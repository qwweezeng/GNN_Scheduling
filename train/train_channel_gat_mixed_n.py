#!/usr/bin/env python3
"""Train one perfect-CSI channel GAT on balanced N=20/30/40/50 graph sizes."""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.train_channel_gat_rician import _output_audit
from train.trainer import Trainer
from utils.artifacts import (
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.project_setup import build_gat_scheduler, load_config


DEFAULT_NODES = (20, 30, 40, 50)
DEFAULT_TRAIN_PER_N = 125
DEFAULT_VAL_PER_N = 32
DEFAULT_TEST_PER_N = 128
SEEDS = {"train": 10086, "val": 10087, "test": 10088}
SIZES = {"train": 300, "val": 128, "test": 128}


def mixed_checkpoint_path(nodes, train_per_n):
    label = "_".join(str(n) for n in nodes)
    return Path("results/models/paper") / (
        f"gat_mixed_N{label}_perN{train_per_n}_"
        "tx=10dBm_csi=perfect_channel=ricianK5dB.pth"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--train-per-n", type=int, default=DEFAULT_TRAIN_PER_N)
    parser.add_argument("--val-per-n", type=int, default=DEFAULT_VAL_PER_N)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--checkpoint-out", default="")
    parser.add_argument("--report-out", default="")
    return parser.parse_args()


def source_config(num_uav):
    config = load_config(ROOT / "configs/paper.yaml")
    config.update({
        "NUM_UAV": int(num_uav),
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    return config


def load_split(num_uav, split, limit):
    config = source_config(num_uav)
    metadata = expected_dataset_metadata(config, num_uav, SEEDS, SIZES)
    train_val_path, test_path = paper_dataset_paths(
        ROOT / "datasets/paper", num_uav, SEEDS, SIZES
    )
    path = test_path if split == "test" else train_val_path
    payload = load_validated_dataset(path, metadata, (split,))
    dataset = payload[f"{split}_set"]
    if limit > len(dataset):
        raise ValueError(
            f"Requested {limit} {split} samples for N={num_uav}; only {len(dataset)} exist"
        )
    return list(dataset[:limit]), {
        "path": str(path),
        "fingerprint": metadata["fingerprint"],
        "available_samples": len(dataset),
        "selected_samples": limit,
    }


def build_balanced_dataset(nodes, split, per_n):
    combined, provenance = [], {}
    for num_uav in nodes:
        subset, info = load_split(num_uav, split, per_n)
        combined.extend(subset)
        provenance[f"N{num_uav}"] = info
    return combined, provenance


def main():
    args = parse_args()
    nodes = tuple(int(value) for value in args.nodes.split(",") if value)
    if not nodes:
        raise ValueError("--nodes cannot be empty")
    if args.train_per_n <= 0 or args.val_per_n <= 0:
        raise ValueError("--train-per-n and --val-per-n must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_set, train_sources = build_balanced_dataset(nodes, "train", args.train_per_n)
    val_set, val_sources = build_balanced_dataset(nodes, "val", args.val_per_n)
    # An equal-sized test subset keeps the post-training integrity audit balanced.
    audit_set, audit_sources = build_balanced_dataset(nodes, "test", DEFAULT_TEST_PER_N)

    config = source_config(nodes[0])
    model = build_gat_scheduler(config, raw_node_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.device in {"auto", "cuda"} and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(
        f"[mixed-train] nodes={nodes}; train={len(train_set)}; "
        f"val={len(val_set)}; audit={len(audit_set)}; device={device}",
        flush=True,
    )
    trainer = Trainer(
        model,
        optimizer,
        device=device,
        gpu_id=args.gpu_id,
        use_queue=False,
        log_dir="logs/channel_gat_mixed_n",
    )
    trained = trainer.train(
        num_epochs=args.epochs,
        train_dataset=train_set,
        val_dataset=val_set,
        eval_interval=1,
        use_queue=False,
    )
    audit = _output_audit(model, audit_set)

    checkpoint_path = (
        Path(args.checkpoint_out)
        if args.checkpoint_out
        else mixed_checkpoint_path(nodes, args.train_per_n)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.report_out)
        if args.report_out
        else Path("results/paper/channel_gat_training") /
        f"rician_channel_gat_mixed_N{'_'.join(map(str, nodes))}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_version": "channel_gat_v1",
        "training_scope": "mixed_network_size",
        "training_nodes": list(nodes),
        "training_composition": {
            f"N{num_uav}": int(args.train_per_n) for num_uav in nodes
        },
        "validation_composition": {
            f"N{num_uav}": int(args.val_per_n) for num_uav in nodes
        },
        "model_state_dict": model.state_dict(),
        "tx_power_dbm": 10.0,
        "csi_mode": "perfect",
        "channel_model": "rician",
        "rician_k_db": 5.0,
        "num_uav": "mixed",
        "source_datasets": {
            "train": train_sources,
            "val": val_sources,
            "audit_test": audit_sources,
        },
        "best_epoch": trained["best_epoch"],
        "best_val_utility": trained["best_val_util"],
        "output_audit": audit,
        "per_epoch_logs": trained["per_epoch_logs"],
        "integrity_status": "passed" if audit["integrity_passed"] else "rejected",
        "timestamp": datetime.now().isoformat(),
    }
    report_path.write_text(
        json.dumps(checkpoint | {"model_state_dict": "omitted"}, indent=2),
        encoding="utf-8",
    )
    if not audit["integrity_passed"]:
        raise RuntimeError(
            "Rejected collapsed mixed-N checkpoint; no model was saved. "
            f"See {report_path}: {audit}"
        )
    torch.save(checkpoint, checkpoint_path)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
        "best_epoch": trained["best_epoch"],
        "best_val_utility": trained["best_val_util"],
        "output_audit": audit,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
