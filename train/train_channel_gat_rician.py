#!/usr/bin/env python3
"""Retrain the conference channel-only GAT on the official Rician baseline."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import calculate_achievable_rate
from train.trainer import Trainer
from utils.io import generate_dataset
from utils.artifacts import (
    atomic_pickle_dump,
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
    write_dataset_manifest,
)
from utils.project_setup import (
    build_gat_scheduler,
    build_network_generator,
    channel_checkpoint_path,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rician_k_db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--val_seed", type=int, default=10087)
    parser.add_argument("--test_seed", type=int, default=10088)
    parser.add_argument("--force_regenerate", action="store_true")
    return parser.parse_args()


def _load_or_generate(config, args):
    seeds = {"train": args.seed, "val": args.val_seed, "test": args.test_seed}
    sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    metadata = expected_dataset_metadata(config, args.num_uav, seeds, sizes)
    train_val_path, test_path = paper_dataset_paths(
        Path("datasets/paper"), args.num_uav, seeds, sizes
    )
    if train_val_path.exists() and test_path.exists() and not args.force_regenerate:
        train_val = load_validated_dataset(
            train_val_path, metadata, ("train", "val")
        )
        test = load_validated_dataset(test_path, metadata, ("test",))
        return (
            train_val["train_set"], train_val["val_set"], test["test_set"],
            train_val_path,
        )
    generator = build_network_generator(config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set = generate_dataset(
        args.train_size, generator, include_queue_feature=False,
        csi_mode="perfect",
    )
    random.seed(args.val_seed)
    np.random.seed(args.val_seed)
    torch.manual_seed(args.val_seed)
    val_set = generate_dataset(
        args.val_size, generator, include_queue_feature=False,
        csi_mode="perfect",
    )
    random.seed(args.test_seed)
    np.random.seed(args.test_seed)
    torch.manual_seed(args.test_seed)
    test_set = generate_dataset(
        args.test_size, generator, include_queue_feature=False,
        csi_mode="perfect",
    )
    atomic_pickle_dump(
        {"metadata": metadata, "train_set": train_set, "val_set": val_set},
        train_val_path,
    )
    atomic_pickle_dump({"metadata": metadata, "test_set": test_set}, test_path)
    write_dataset_manifest(
        train_val_path.parent, metadata, train_val_path, test_path
    )
    return train_set, val_set, test_set, train_val_path


def _output_audit(model, dataset):
    probabilities, binary_counts, link_counts = [], [], []
    hard_utilities, all_active_utilities = [], []
    model.eval()
    with torch.no_grad():
        for state in dataset:
            pi, _ = model(
                state.vertex_features[:, :1], state.edge_index, state.edge_weights
            )
            value = pi.reshape(-1).cpu()
            hard = (value >= 0.5).float().unsqueeze(1)
            probabilities.append(value)
            binary_counts.append(float(hard.sum()))
            link_counts.append(float(value.numel()))
            hard_utilities.append(float(
                calculate_achievable_rate(
                    hard, state, binary=False
                ).sum().item()
            ))
            all_active_utilities.append(float(
                calculate_achievable_rate(
                    torch.ones_like(hard), state, binary=False
                ).sum().item()
            ))
    values = torch.cat(probabilities)
    all_active_fraction = float(np.mean([
        active == total for active, total in zip(binary_counts, link_counts)
    ]))
    all_inactive_fraction = float(np.mean([
        active == 0 for active in binary_counts
    ]))
    hard_utility = float(np.mean(hard_utilities))
    all_active_utility = float(np.mean(all_active_utilities))
    integrity_passed = bool(
        all_active_fraction < 0.95
        and all_inactive_fraction < 0.95
        and hard_utility > 1.05 * all_active_utility
    )
    return {
        "probability_mean": float(values.mean()),
        "probability_std": float(values.std(unbiased=False)),
        "probability_min": float(values.min()),
        "probability_max": float(values.max()),
        "mean_binary_active_fraction": float(
            np.mean(np.asarray(binary_counts) / np.asarray(link_counts))
        ),
        "mean_binary_active_links": float(np.mean(binary_counts)),
        "all_links_active_state_fraction": all_active_fraction,
        "all_links_inactive_state_fraction": all_inactive_fraction,
        "hard_test_utility": hard_utility,
        "all_active_test_utility": all_active_utility,
        "hard_gain_over_all_active": hard_utility - all_active_utility,
        "integrity_rule": (
            "all-active states < 95%, all-inactive states < 95%, and hard "
            "utility > 1.05 * all-active utility"
        ),
        "integrity_passed": integrity_passed,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": args.rician_k_db,
        "CSI_MODE": "perfect",
    })
    train_set, val_set, test_set, dataset_path = _load_or_generate(config, args)
    dataset_metadata = expected_dataset_metadata(
        config,
        args.num_uav,
        {"train": args.seed, "val": args.val_seed, "test": args.test_seed},
        {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        },
    )
    model = build_gat_scheduler(config, raw_node_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(
        model, optimizer, device="cpu", use_queue=False,
        log_dir="logs/channel_gat_rician",
    )
    trained = trainer.train(
        num_epochs=args.epochs,
        train_dataset=train_set,
        val_dataset=val_set,
        eval_interval=1,
        use_queue=False,
    )
    test_utility = float(trainer.evaluate(model, test_set, use_queue=False))
    audit = _output_audit(model, test_set)
    checkpoint_path = channel_checkpoint_path(config, args.num_uav)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_version": "channel_gat_v1",
        "model_state_dict": model.state_dict(),
        "tx_power_dbm": float(config.get("TX_POWER", 10.0)),
        "csi_mode": "perfect",
        "channel_model": "rician",
        "rician_k_db": float(args.rician_k_db),
        "num_uav": int(args.num_uav),
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": dataset_metadata["fingerprint"],
        "config": {
            "NUM_UAV": int(args.num_uav),
            "TX_POWER": float(config["TX_POWER"]),
            "CHANNEL_MODEL": "rician",
            "RICIAN_K_DB": float(args.rician_k_db),
            "CSI_MODE": "perfect",
            "TRAIN_SIZE": int(args.train_size),
            "VAL_SIZE": int(args.val_size),
            "TEST_SIZE": int(args.test_size),
            "SEED": int(args.seed),
        },
        "best_epoch": trained["best_epoch"],
        "best_val_utility": trained["best_val_util"],
        "test_utility": test_utility,
        "output_audit": audit,
        "per_epoch_logs": trained["per_epoch_logs"],
        "integrity_status": "passed" if audit["integrity_passed"] else "rejected",
        "timestamp": datetime.now().isoformat(),
    }
    report_path = Path("results/paper/channel_gat_training") / (
        f"rician_channel_gat_N{args.num_uav}_K{args.rician_k_db:g}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        key: value for key, value in checkpoint.items()
        if key not in {"model_state_dict"}
    }, indent=2), encoding="utf-8")
    if not audit["integrity_passed"]:
        raise RuntimeError(
            "Rejected collapsed channel-only GAT; the production checkpoint "
            f"was not overwritten. See {report_path}: {audit}"
        )
    torch.save(checkpoint, checkpoint_path)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
        "test_utility": test_utility,
        "output_audit": audit,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
