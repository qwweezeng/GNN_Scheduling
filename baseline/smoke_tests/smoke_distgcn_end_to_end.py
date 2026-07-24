#!/usr/bin/env python3
"""One real UAV state through GCN inference, LGS, mapping, and SINR settlement."""

import json
import argparse
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from baseline.distgcn_adapter import DistGCNQueueScheduler  # noqa: E402
from baseline.queue_schedulers import HardConflictLocalGreedyScheduler  # noqa: E402
from models.model import calculate_achievable_rate  # noqa: E402
from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.feasibility import shared_uav_action_audit  # noqa: E402
from utils.project_setup import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/paper/distgcn_adapter/end_to_end_smoke.json",
    )
    args = parser.parse_args()
    config = load_config("configs/paper.yaml")
    config.update({
        "NUM_UAV": 20,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 500, "val": 128, "test": 128}
    metadata = expected_dataset_metadata(config, 20, seeds, sizes)
    _, test_path = paper_dataset_paths(REPO / "datasets/paper", 20, seeds, sizes)
    state = load_validated_dataset(
        test_path, metadata, ("test",)
    )["test_set"][0]
    queues = torch.arange(1, state.num_vertices + 1, dtype=torch.float32)
    scheduler = DistGCNQueueScheduler()
    action = scheduler.schedule_with_state(state, queues).reshape(-1)
    audit = shared_uav_action_audit(action, state.links)
    if not audit["feasible"]:
        raise RuntimeError("DistGCN end-to-end action violates shared-UAV feasibility")
    rate = calculate_achievable_rate(
        action.reshape(-1, 1), state, binary=True, detach_hard=True
    ).sum()
    lgs = HardConflictLocalGreedyScheduler()
    lgs_action = lgs.schedule_with_state(state, queues).reshape(-1)
    lgs_rate = calculate_achievable_rate(
        lgs_action.reshape(-1, 1), state, binary=True, detach_hard=True
    ).sum()
    payload = {
        "status": "pass",
        "num_links": int(state.num_vertices),
        "active_links": int(action.sum()),
        "aggregate_phy_rate_mbps": float(rate) * float(state.bandwidth) / 1e6,
        "hard_conflict_lgs_active_links": int(lgs_action.sum()),
        "hard_conflict_lgs_phy_rate_mbps": (
            float(lgs_rate) * float(state.bandwidth) / 1e6
        ),
        "hard_feasibility": audit,
        "distgcn": scheduler.last_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
