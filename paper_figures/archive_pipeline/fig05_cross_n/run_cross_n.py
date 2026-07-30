#!/usr/bin/env python3
"""Historical Fig. 5 cross-network evaluation runner.

Restored for traceability of the previous pipeline.  The active, maintained
implementation is ``experiments/evaluate_channel_gat_cross_n.py`` (current
Fig. 6); this file deliberately retains the old directory layout and schema.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.model import calculate_achievable_rate
from utils.artifacts import (
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.feasibility import decode_endpoint_mutual_matching, shared_uav_action_audit
from utils.project_setup import load_channel_gat, load_config


def direct_rate_priority(pi, state):
    tx, rx = state.links[:, 0], state.links[:, 1]
    snr = state.signal_power[tx, rx].cpu() / state.noise_power
    return pi.reshape(-1).float().cpu() * torch.log2(
        1.0 + snr / state.snr_gap
    ).float()


def cfg(num_uav):
    config = load_config(ROOT / "configs/paper.yaml")
    config.update({
        "NUM_UAV": num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    return config


def states(num_uav):
    config = cfg(num_uav)
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 300, "val": 128, "test": 128}
    metadata = expected_dataset_metadata(config, num_uav, seeds, sizes)
    _, test_path = paper_dataset_paths(ROOT / "datasets/paper", num_uav, seeds, sizes)
    return (
        load_validated_dataset(test_path, metadata, ("test",))["test_set"],
        metadata["fingerprint"],
        str(test_path),
    )


def evaluate(model, test_set):
    rates, active, audit = [], [], []
    with torch.no_grad():
        for index, state in enumerate(test_set):
            device = next(model.parameters()).device
            pi, _ = model(
                state.vertex_features[:, :1].to(device),
                state.edge_index.to(device),
                state.edge_weights.to(device),
            )
            candidate = (pi >= 0.5).float().cpu()
            action, decoder_audit = decode_endpoint_mutual_matching(
                candidate,
                state.links,
                priorities=direct_rate_priority(pi, state),
                max_rounds=1,
            )
            feasibility = shared_uav_action_audit(action, state.links)
            if not feasibility["feasible"]:
                raise RuntimeError("infeasible decoded action")
            rate = float(calculate_achievable_rate(
                action.reshape(-1, 1), state, binary=True, detach_hard=True
            ).sum())
            rates.append(rate)
            active.append(float(action.float().mean()))
            audit.append({
                "sample_index": index,
                "active_project_link_indices": torch.nonzero(
                    action.reshape(-1) > 0.5, as_tuple=True
                )[0].tolist(),
                "hard_feasibility": decoder_audit,
            })
    values = np.asarray(rates)
    return {
        "mean_sum_spectral_efficiency_bit_s_hz": float(values.mean()),
        "sem_sum_spectral_efficiency_bit_s_hz": float(
            values.std(ddof=1) / np.sqrt(len(values))
        ),
        "mean_aggregate_phy_rate_mbps": float(values.mean() * 5),
        "mean_active_fraction": float(np.mean(active)),
        "per_sample_actions": audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    nodes = [int(value) for value in args.nodes.split(",")]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )

    test_sets = {num_uav: states(num_uav) for num_uav in nodes}
    matrix, checkpoints = {}, {}
    for train_num_uav in nodes:
        model, checkpoint = load_channel_gat(
            cfg(train_num_uav), train_num_uav, device=device
        )
        checkpoints[str(train_num_uav)] = checkpoint
        matrix[str(train_num_uav)] = {}
        for test_num_uav in nodes:
            result = evaluate(model, test_sets[test_num_uav][0])
            matrix[str(train_num_uav)][str(test_num_uav)] = result

    dedicated = {
        str(num_uav): matrix[str(num_uav)][str(num_uav)][
            "mean_aggregate_phy_rate_mbps"
        ]
        for num_uav in nodes
    }
    relative = {
        train_num_uav: {
            test_num_uav: value["mean_aggregate_phy_rate_mbps"] /
            dedicated[test_num_uav]
            for test_num_uav, value in row.items()
        }
        for train_num_uav, row in matrix.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "tnse_fig05_cross_n_v1",
        "nodes": nodes,
        "physics": {
            "tx_power_dbm": 10,
            "channel_model": "rician",
            "rician_k_db": 5,
            "service_channel": "current true instantaneous Rician channel",
            "decoder": "one_round_endpoint_mutual_matching",
        },
        "checkpoints": checkpoints,
        "test_sets": {
            str(num_uav): {
                "path": test_sets[num_uav][2],
                "fingerprint": test_sets[num_uav][1],
                "samples": len(test_sets[num_uav][0]),
            }
            for num_uav in nodes
        },
        "results": matrix,
        "relative_to_dedicated_test_model": relative,
    }, indent=2))


if __name__ == "__main__":
    main()
