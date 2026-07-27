#!/usr/bin/env python3
"""Paper Fig. 1: full-buffer proposed GAT versus retained baselines.

The script reads the canonical Rician test split and a metadata-checked GAT
checkpoint.  Every returned schedule is settled by the project's Shannon-rate
function.  Missing third-party baselines fail loudly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model import calculate_achievable_rate  # noqa: E402
from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.baseline import (  # noqa: E402
    eval_spatial_dl,
    fplinq_schedule,
    local_greedy,
    random_activate,
    simple_greedy,
)
from models.local_physical_reuse import select_local_maxima  # noqa: E402
from utils.feasibility import (  # noqa: E402
    decode_endpoint_mutual_matching,
    project_shared_uav_feasible,
    shared_uav_action_audit,
)
from utils.project_setup import (  # noqa: E402
    load_channel_gat,
    load_channel_gat_checkpoint,
    load_config,
)


def _canonical_test(config, num_uav, test_size):
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 300, "val": 128, "test": 128}
    expected = expected_dataset_metadata(config, num_uav, seeds, sizes)
    _, test_path = paper_dataset_paths(Path("datasets/paper"), num_uav, seeds, sizes)
    payload = load_validated_dataset(test_path, expected, ("test",))
    if int(test_size) > len(payload["test_set"]):
        raise ValueError(
            f"Requested {test_size} samples, canonical test split has "
            f"{len(payload['test_set'])}"
        )
    return payload["test_set"][: int(test_size)], test_path, expected["fingerprint"]


def _project_and_settle(actions, priorities, states, decoder="global_priority"):
    utilities, fractions, sample_records = [], [], []
    for sample_index, (action, priority, state) in enumerate(
        zip(actions, priorities, states)
    ):
        if decoder == "endpoint_mutual_1":
            action, feasibility = decode_endpoint_mutual_matching(
                action, state.links, priorities=priority, max_rounds=1
            )
        elif decoder == "global_priority":
            action, feasibility = project_shared_uav_feasible(
                action, state.links, priorities=priority
            )
        else:
            raise ValueError(f"unknown hard-feasibility decoder: {decoder}")
        action = action.to(dtype=torch.float32).reshape(-1, 1)
        audit = shared_uav_action_audit(action, state.links)
        if not audit["feasible"]:
            raise RuntimeError(f"{decoder} emitted an infeasible action")
        utilities.append(float(calculate_achievable_rate(
            action, state, binary=True, detach_hard=True
        ).sum()))
        fractions.append(float(action.mean()))
        active = torch.nonzero(action.reshape(-1), as_tuple=True)[0].tolist()
        aggregate_phy_rate_mbps = utilities[-1] * float(state.bandwidth) / 1e6
        sample_records.append({
            "sample_index": sample_index,
            "num_links": int(state.links.size(0)),
            "sum_spectral_efficiency_bit_s_hz": utilities[-1],
            "aggregate_phy_rate_mbps": aggregate_phy_rate_mbps,
            "active_fraction": fractions[-1],
            "active_project_link_indices": active,
            "hard_feasibility": feasibility,
        })
    return utilities, fractions, sample_records


def _direct_rate_priority(probability, state):
    tx, rx = state.links[:, 0], state.links[:, 1]
    direct_snr = state.signal_power[tx, rx].cpu() / state.noise_power
    direct_rate = torch.log2(1.0 + direct_snr / state.snr_gap)
    return probability.reshape(-1).float().cpu() * direct_rate.float()


def _gat_actions(model, states):
    actions, priorities = [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for state in states:
            probability, _ = model(
                state.vertex_features[:, :1].to(device),
                state.edge_index.to(device),
                state.edge_weights.to(device),
            )
            actions.append((probability >= 0.5).float().cpu())
            priorities.append(_direct_rate_priority(probability, state))
    return actions, priorities


def _snr_localmax_actions(states):
    """One-hop distributed LocalMax using direct-SNR priorities."""
    actions, priorities = [], []
    for state in states:
        snr = state.vertex_features[:, 0].detach().cpu()
        actions.append(select_local_maxima(
            snr, torch.ones_like(snr, dtype=torch.bool), state.edge_index
        ).float())
        priorities.append(snr)
    return actions, priorities


def _summary(values, active, sample_records):
    array = np.asarray(values, dtype=float)
    active_array = np.asarray(active, dtype=float)
    phy_rate = np.asarray(
        [item["aggregate_phy_rate_mbps"] for item in sample_records], dtype=float
    )
    return {
        "mean_sum_spectral_efficiency": float(array.mean()),
        "std_sum_spectral_efficiency": float(array.std(ddof=0)),
        "standard_error": float(array.std(ddof=1) / np.sqrt(len(array)))
        if len(array) > 1 else 0.0,
        "mean_aggregate_phy_rate_mbps": float(phy_rate.mean()),
        "std_aggregate_phy_rate_mbps": float(phy_rate.std(ddof=0)),
        "standard_error_aggregate_phy_rate_mbps": float(
            phy_rate.std(ddof=1) / np.sqrt(len(phy_rate))
        ) if len(phy_rate) > 1 else 0.0,
        "mean_active_fraction": (
            float(np.nanmean(active_array))
            if np.isfinite(active_array).any() else None
        ),
        "per_sample": array.tolist(),
        "per_sample_records": sample_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--gat_model", default="")
    parser.add_argument("--seed", type=int, default=10088)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
        help="GAT inference device; auto selects CUDA when available.",
    )
    parser.add_argument("--fplinq_iterations", type=int, default=50)
    parser.add_argument(
        "--top_snr_ratio", type=float, default=0.02,
        help="Fixed global Top-SNR candidate fraction; Fig. 1 uses 2%%.",
    )
    parser.add_argument(
        "--methods",
        default=(
            "Random,Top-SNR,Local Greedy,Spatial DL,FPLinQ,"
            "Proposed-GAT"
        ),
    )
    parser.add_argument(
        "--output", default="results/paper/full_buffer/benchmark.json"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[evaluate] GAT device={device}", flush=True)
    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    states, dataset_path, fingerprint = _canonical_test(
        config, args.num_uav, args.num_samples
    )
    requested = [name.strip() for name in args.methods.split(",") if name.strip()]
    results = {}
    method_metadata = {}

    for name in requested:
        decoder = "global_priority"
        if name == "Random":
            torch.manual_seed(args.seed + 1)
            actions, priorities = [], []
            for state in states:
                actions.append(random_activate(state, 0.05))
                priorities.append(torch.rand(int(state.links.size(0))))
            priority_rule = "independent_seeded_random_order"
        elif name == "Top-SNR":
            actions, priorities = [], []
            for state in states:
                action = simple_greedy(state, float(args.top_snr_ratio))
                actions.append(action)
                priorities.append(state.vertex_features[:, 0].detach().cpu())
            priority_rule = f"global_direct_snr_top_{args.top_snr_ratio:g}_fraction"
            method_metadata[name] = {
                "global_top_k": True,
                "top_snr_ratio": float(args.top_snr_ratio),
                "coordination": "centralized global direct-SNR ordering",
            }
        elif name == "LocalMax":
            actions, priorities = _snr_localmax_actions(states)
            priority_rule = "one_hop_direct_snr_local_maximum"
            method_metadata[name] = {
                "coordination": "one-hop conflict-neighbor SNR exchange",
                "global_top_k": False,
            }
        elif name == "Local Greedy":
            actions = [
                local_greedy(state, interference_threshold=0.3)
                for state in states
            ]
            priorities = [state.vertex_features[:, 0] for state in states]
            priority_rule = "observed_direct_link_snr"
        elif name == "Spatial DL":
            _, actions, spatial_metadata = eval_spatial_dl(
                states,
                seed=args.seed,
                return_actions=True,
                return_metadata=True,
            )
            priorities = [state.vertex_features[:, 0] for state in states]
            priority_rule = (
                "observed_direct_link_snr_only_for_resolving_shared_uav_"
                "collisions_in_external_checkpoint_action"
            )
            method_metadata[name] = {
                "checkpoint_training_link_count": 50,
                "cardinality_policy": "full_link_dynamic_placeholders",
                "training_boundary": (
                    "original 50-link checkpoint; zero-shot cardinality and "
                    "Rician-UAV distribution transfer"
                ),
                "per_sample_adapter": spatial_metadata,
            }
        elif name == "FPLinQ":
            actions, priorities = [], []
            for state in states:
                action, allocation = fplinq_schedule(
                    state,
                    use_fading=True,
                    iterations=args.fplinq_iterations,
                )
                actions.append(action)
                priorities.append(allocation)
            priority_rule = "continuous_fplinq_allocation"
            method_metadata[name] = {
                "iterations": int(args.fplinq_iterations),
                "csi_scope": "full_received_power_matrix",
                "training": "none",
            }
        elif name == "Proposed-GAT":
            if args.gat_model:
                model, checkpoint_path = load_channel_gat_checkpoint(
                    config,
                    args.gat_model,
                    args.num_uav,
                    device=device,
                    expected_dataset_fingerprint=fingerprint,
                )
            else:
                model, checkpoint_path = load_channel_gat(
                    config,
                    args.num_uav,
                    device=device,
                    expected_dataset_fingerprint=fingerprint,
                )
            actions, priorities = _gat_actions(model, states)
            priority_rule = "proposed_gat_probability_x_direct_rate"
            decoder = "endpoint_mutual_1"
            method_metadata[name] = {
                "checkpoint": checkpoint_path,
                "feasible_decoder": "one_round_endpoint_mutual_handshake",
                "decoder_training": "none; deterministic score decoder",
                "gat_priority_mode": "probability_x_direct_rate",
            }
        else:
            raise ValueError(f"Unknown method: {name}")
        values, fractions, sample_records = _project_and_settle(
            actions, priorities, states, decoder=decoder
        )
        projection_modified = sum(
            record["hard_feasibility"]["candidate"]["active_links"]
            != record["hard_feasibility"]["projected"]["active_links"]
            for record in sample_records
        )
        method_metadata.setdefault(name, {})
        method_metadata[name].update({
            "variant": name,
            "hard_feasibility": "single_radio_half_duplex_matching",
            "projection_priority": priority_rule,
            "feasibility_decoder": decoder,
            "states_modified_by_projection": int(projection_modified),
        })
        results[name] = _summary(values, fractions, sample_records)
        print(
            f"{name}: {results[name]['mean_sum_spectral_efficiency']:.4f} "
            "bit/s/Hz, "
            f"{results[name]['mean_aggregate_phy_rate_mbps']:.4f} Mbit/s"
        )

    output = Path(args.output)
    if output.name == "benchmark.json" and args.num_uav != 20:
        output = output.with_name(f"benchmark_N{args.num_uav}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "paper_full_buffer_benchmark_v1",
        "definition": "all actions settled on the canonical perfect Rician channel",
        "metric_definition": {
            "sum_spectral_efficiency_bit_s_hz": "sum_i log2(1+SINR_i/Gamma)",
            "aggregate_phy_rate_mbps": (
                "bandwidth_hz * sum_spectral_efficiency_bit_s_hz / 1e6"
            ),
            "not_queue_utility": True,
            "not_packet_goodput": True,
        },
        "num_uav": args.num_uav,
        "num_samples": len(states),
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": fingerprint,
        "seed": int(args.seed),
        "shared_uav_constraint": (
            "hard single-radio/half-duplex matching applied to every method"
        ),
        "service_channel": "current true instantaneous Rician channel",
        "config": config,
        "method_metadata": method_metadata,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
