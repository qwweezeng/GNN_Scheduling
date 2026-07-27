#!/usr/bin/env python3
"""Evaluate a channel-GAT checkpoint without endpoint-conflict projection.

This diagnostic reports the utility of direct 0.5-thresholded actions. Such
actions can violate the shared-UAV matching constraint and are not schedulable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model import calculate_achievable_rate
from utils.artifacts import expected_dataset_metadata, load_validated_dataset, paper_dataset_paths
from utils.feasibility import shared_uav_action_audit
from utils.project_setup import build_gat_scheduler, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-uav", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"[evaluate-no-projection] GAT device={device}", flush=True)

    config = load_config(ROOT / "configs/paper.yaml")
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 300, "val": 128, "test": 128}
    metadata = expected_dataset_metadata(config, args.num_uav, seeds, sizes)
    _, dataset_path = paper_dataset_paths(ROOT / "datasets/paper", args.num_uav, seeds, sizes)
    states = load_validated_dataset(dataset_path, metadata, ("test",))["test_set"]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_gat_scheduler(config, raw_node_dim=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rates_mbps: list[float] = []
    audits: list[dict[str, object]] = []
    with torch.no_grad():
        for state in states:
            pi, _ = model(
                state.vertex_features[:, :1].to(device),
                state.edge_index.to(device),
                state.edge_weights.to(device),
            )
            action = (pi >= 0.5).float().cpu()
            audit = shared_uav_action_audit(action, state.links)
            rates_mbps.append(
                float(
                    calculate_achievable_rate(
                        action.reshape(-1, 1), state, binary=True, detach_hard=True
                    ).sum()
                )
                * float(state.bandwidth)
                / 1e6
            )
            audits.append(audit)

    output = {
        "num_uav": args.num_uav,
        "checkpoint": args.checkpoint,
        "canonical_test_fingerprint": metadata["fingerprint"],
        "num_test_samples": len(rates_mbps),
        "decision_rule": "direct_threshold_pi_ge_0.5_without_endpoint_projection",
        "schedulable": False,
        "test_unprojected_threshold_utility_mbps": float(np.mean(rates_mbps)),
        "test_unprojected_threshold_utility_sem_mbps": float(
            np.std(rates_mbps, ddof=1) / np.sqrt(len(rates_mbps))
        ),
        "infeasible_state_fraction": float(np.mean([not audit["feasible"] for audit in audits])),
        "mean_active_links": float(np.mean([audit["active_links"] for audit in audits])),
        "mean_active_link_fraction": float(np.mean([audit["active_fraction"] for audit in audits])),
        "mean_violating_uavs": float(np.mean([audit["num_violating_uavs"] for audit in audits])),
        "mean_endpoint_excess_count": float(
            np.mean([audit["endpoint_excess_count"] for audit in audits])
        ),
        "mean_max_active_links_per_uav": float(
            np.mean([audit["max_active_links_per_uav"] for audit in audits])
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
