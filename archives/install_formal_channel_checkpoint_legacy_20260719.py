#!/usr/bin/env python3
"""Install an audited formal channel-GAT checkpoint as the canonical model."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    report_path = Path(args.report)
    destination = Path(args.destination)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)
    settings = report["settings"]
    if (
        settings["num_uav"] != 20
        or settings["channel_model"] != "rician"
        or settings["tx_power_dbm"] != 10
        or settings["rician_k_db"] != 5
    ):
        raise ValueError(f"Refusing to install nonformal settings: {settings}")
    audit = report["transfer"]["10"]["output_audit"]
    if not audit["integrity_passed"]:
        raise ValueError(f"Refusing failed output audit: {audit}")
    payload = {
        "model_version": "channel_gat_v1",
        "model_state_dict": checkpoint["model_state_dict"],
        "tx_power_dbm": 10.0,
        "csi_mode": "perfect",
        "channel_model": "rician",
        "rician_k_db": 5.0,
        "num_uav": 20,
        "dataset_path": (
            "datasets/formal/channel_gat_power_transfer/N20/rician/"
            "train_tx=10dBm_seed=10086_n=500-128.pkl"
        ),
        "test_dataset_path": (
            "datasets/formal/channel_gat_power_transfer/N20/rician/"
            "test_tx=10dBm_seed=10088_n=128.pkl"
        ),
        "best_epoch": report["best_epoch"],
        "best_val_utility": report["best_val_utility"],
        "test_utility": audit["hard_test_utility"],
        "output_audit": audit,
        "integrity_status": "accepted",
        "source_checkpoint": str(checkpoint_path),
        "source_report": str(report_path),
        "selection_rule": "matched N20 Rician K5dB train10-test10",
        "timestamp": report["timestamp"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Installed canonical checkpoint: {destination}")


if __name__ == "__main__":
    main()
