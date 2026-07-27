#!/usr/bin/env python3
"""Evaluate a channel-GAT checkpoint with endpoint-matching hard feasibility."""
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
from utils.feasibility import decode_endpoint_mutual_matching, shared_uav_action_audit
from utils.project_setup import build_gat_scheduler, load_config


def _direct_rate_priority(pi: torch.Tensor, state) -> torch.Tensor:
    tx, rx = state.links[:, 0], state.links[:, 1]
    snr = state.signal_power[tx, rx].cpu() / state.noise_power
    return pi.reshape(-1).float().cpu() * torch.log2(1.0 + snr / state.snr_gap).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-uav", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.device in {"auto", "cuda"} and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[evaluate-projected] GAT device={device}", flush=True)

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
    with torch.no_grad():
        for state in states:
            pi, _ = model(
                state.vertex_features[:, :1].to(device),
                state.edge_index.to(device),
                state.edge_weights.to(device),
            )
            action, _ = decode_endpoint_mutual_matching(
                (pi >= 0.5).float().cpu(),
                state.links,
                priorities=_direct_rate_priority(pi, state),
                max_rounds=1,
            )
            if not shared_uav_action_audit(action, state.links)["feasible"]:
                raise RuntimeError("infeasible decoded action")
            rate = calculate_achievable_rate(
                action.reshape(-1, 1), state, binary=True, detach_hard=True
            ).sum()
            rates_mbps.append(float(rate) * float(state.bandwidth) / 1e6)

    output = {
        "num_uav": args.num_uav,
        "checkpoint": args.checkpoint,
        "canonical_test_fingerprint": metadata["fingerprint"],
        "num_test_samples": len(rates_mbps),
        "decision_rule": "hard_threshold_pi_ge_0.5_endpoint_mutual_matching",
        "schedulable": True,
        "test_projected_binary_rate_mbps": float(np.mean(rates_mbps)),
        "test_projected_binary_rate_sem_mbps": float(
            np.std(rates_mbps, ddof=1) / np.sqrt(len(rates_mbps))
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
