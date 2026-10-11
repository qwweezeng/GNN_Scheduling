#!/usr/bin/env python3
"""Evaluate two catalog-pilot GATs on both paired test catalogs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import calculate_achievable_rate
from utils.artifacts import (
    expected_dataset_metadata_quantized,
    load_validated_dataset,
    paper_dataset_paths_quantized,
)
from utils.baseline import local_greedy
from utils.feasibility import (
    decode_endpoint_mutual_matching,
    project_shared_uav_feasible,
    shared_uav_action_audit,
)
from utils.project_setup import build_gat_scheduler, load_config


POLICIES = ("instantaneous", "mean_power")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-uav", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--dataset-root", default="datasets/catalog_pilot")
    parser.add_argument("--instantaneous-checkpoint")
    parser.add_argument("--mean-power-checkpoint")
    parser.add_argument(
        "--data-only", action="store_true",
        help="Audit the paired test catalogs without loading trained models.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def load_test_sets(config, args):
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    result = {}
    fingerprints = {}
    paths = {}
    for policy in POLICIES:
        metadata = expected_dataset_metadata_quantized(
            config, args.num_uav, seeds, sizes, "perfect", None,
            catalog_policy=policy,
        )
        _, test_path = paper_dataset_paths_quantized(
            Path(args.dataset_root), args.num_uav, seeds, sizes,
            "perfect", None, catalog_policy=policy,
        )
        result[policy] = load_validated_dataset(
            test_path, metadata, ("test",)
        )["test_set"]
        fingerprints[policy] = metadata["fingerprint"]
        paths[policy] = str(test_path)
    return result, fingerprints, paths


def paired_catalog_summary(test_sets):
    instantaneous = test_sets["instantaneous"]
    mean_power = test_sets["mean_power"]
    rows = []
    for left, right in zip(instantaneous, mean_power):
        if not torch.equal(left.positions, right.positions):
            raise RuntimeError("paired test positions differ")
        if not torch.equal(left.signal_power, right.signal_power):
            raise RuntimeError("paired instantaneous power matrices differ")
        left_links = {tuple(link) for link in left.links.tolist()}
        right_links = {tuple(link) for link in right.links.tolist()}
        union = left_links | right_links
        rows.append({
            "instantaneous_links": len(left_links),
            "mean_power_links": len(right_links),
            "jaccard": len(left_links & right_links) / len(union),
            "symmetric_difference": len(left_links ^ right_links),
            "instantaneous_edges": int(left.edge_index.shape[1]),
            "mean_power_edges": int(right.edge_index.shape[1]),
        })
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def direct_rate_priority(pi, state):
    tx, rx = state.links[:, 0], state.links[:, 1]
    snr = state.signal_power[tx, rx].cpu() / state.noise_power
    return pi.reshape(-1).float().cpu() * torch.log2(
        1.0 + snr / state.snr_gap
    ).float()


def load_model(config, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = build_gat_scheduler(config, raw_node_dim=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint


def evaluate_model(model, states, device):
    rates, active_counts, probability_means = [], [], []
    all_inactive = 0
    with torch.no_grad():
        for state in states:
            pi, _ = model(
                state.vertex_features[:, :1].to(device),
                state.edge_index.to(device),
                state.edge_weights.to(device),
            )
            candidate = (pi >= 0.5).float().cpu()
            action, _ = decode_endpoint_mutual_matching(
                candidate, state.links,
                priorities=direct_rate_priority(pi, state),
                max_rounds=1,
            )
            if not shared_uav_action_audit(action, state.links)["feasible"]:
                raise RuntimeError("infeasible endpoint-mutual action")
            rate = calculate_achievable_rate(
                action.reshape(-1, 1), state,
                binary=True, detach_hard=True,
            ).sum()
            rates.append(float(rate) * float(state.bandwidth) / 1e6)
            active_counts.append(float(action.sum()))
            probability_means.append(float(pi.mean()))
            all_inactive += int(float(action.sum()) == 0.0)
    return {
        "mean_rate_mbps": float(np.mean(rates)),
        "rate_sem_mbps": float(np.std(rates, ddof=1) / np.sqrt(len(rates))),
        "mean_active_links": float(np.mean(active_counts)),
        "mean_probability": float(np.mean(probability_means)),
        "all_inactive_sample_fraction": all_inactive / len(states),
        "sample_rates_mbps": rates,
    }


def evaluate_local_greedy(states):
    rates, active_counts, rejected_counts = [], [], []
    for state in states:
        candidate = local_greedy(state, interference_threshold=0.3)
        priorities = direct_rate_priority(
            torch.ones(state.links.size(0)), state
        )
        action, projection = project_shared_uav_feasible(
            candidate, state.links, priorities=priorities
        )
        if not shared_uav_action_audit(action, state.links)["feasible"]:
            raise RuntimeError("projected local greedy produced infeasible action")
        rate = calculate_achievable_rate(
            action.reshape(-1, 1), state, binary=True
        ).sum()
        rates.append(float(rate) * float(state.bandwidth) / 1e6)
        active_counts.append(float(action.sum()))
        rejected_counts.append(len(projection["rejected_project_link_indices"]))
    return {
        "method": "local_greedy_plus_shared_uav_projection",
        "mean_rate_mbps": float(np.mean(rates)),
        "rate_sem_mbps": float(np.std(rates, ddof=1) / np.sqrt(len(rates))),
        "mean_active_links": float(np.mean(active_counts)),
        "mean_projection_rejections": float(np.mean(rejected_counts)),
    }


def main():
    args = parse_args()
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.device in {"auto", "cuda"} and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    config = load_config("configs/paper.yaml")
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
        "Q_BITS": None,
    })
    test_sets, fingerprints, paths = load_test_sets(config, args)
    output = {
        "experiment": "paired_catalog_pilot_v1",
        "num_uav": args.num_uav,
        "num_test_samples": args.test_size,
        "device": str(device),
        "test_dataset_fingerprints": fingerprints,
        "test_dataset_paths": paths,
        "paired_catalog_summary": paired_catalog_summary(test_sets),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.data_only:
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2), flush=True)
        return
    if not args.instantaneous_checkpoint or not args.mean_power_checkpoint:
        raise ValueError("both checkpoint arguments are required for model evaluation")
    checkpoints = {
        "instantaneous_trained": args.instantaneous_checkpoint,
        "mean_power_trained": args.mean_power_checkpoint,
    }
    models = {}
    checkpoint_metadata = {}
    for label, checkpoint_path in checkpoints.items():
        models[label], checkpoint = load_model(config, checkpoint_path, device)
        checkpoint_metadata[label] = {
            "path": checkpoint_path,
            "catalog_policy": checkpoint.get("catalog_policy"),
            "dataset_fingerprint": checkpoint.get("dataset_fingerprint"),
            "best_epoch": checkpoint.get("best_epoch"),
            "integrity_status": checkpoint.get("integrity_status"),
        }

    results = {}
    sample_rates = {}
    references = {}
    for policy, states in test_sets.items():
        references[policy] = evaluate_local_greedy(states)
        for model_label, model in models.items():
            result = evaluate_model(model, states, device)
            result["rate_relative_to_local_greedy"] = (
                result["mean_rate_mbps"]
                / references[policy]["mean_rate_mbps"]
            )
            key = f"{model_label}_on_{policy}"
            sample_rates[key] = result.pop("sample_rates_mbps")
            results[key] = result

    matched_instantaneous = np.asarray(
        sample_rates["instantaneous_trained_on_instantaneous"]
    )
    matched_mean_power = np.asarray(
        sample_rates["mean_power_trained_on_mean_power"]
    )
    paired_difference = matched_instantaneous - matched_mean_power
    paired_sem = float(
        paired_difference.std(ddof=1) / np.sqrt(len(paired_difference))
    )
    paired_comparison = {
        "difference_definition": "instantaneous_minus_mean_power_mbps",
        "mean_difference_mbps": float(paired_difference.mean()),
        "difference_sem_mbps": paired_sem,
        "normal_approx_95pct_ci_mbps": [
            float(paired_difference.mean() - 1.96 * paired_sem),
            float(paired_difference.mean() + 1.96 * paired_sem),
        ],
        "instantaneous_win_fraction": float(np.mean(paired_difference > 0)),
        "tie_fraction": float(np.mean(paired_difference == 0)),
        "relative_mean_difference": float(
            paired_difference.mean() / matched_mean_power.mean()
        ),
    }

    output.update({
        "checkpoints": checkpoint_metadata,
        "local_greedy_reference": references,
        "cross_evaluation": results,
        "matched_catalog_paired_comparison": paired_comparison,
    })
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
