#!/usr/bin/env python3
"""Paper Fig. 3: train-size/test-size generalization of Channel-GAT."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.train_channel_gat_rician import _output_audit  # noqa: E402
from train.trainer import Trainer  # noqa: E402
from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.project_setup import (  # noqa: E402
    build_gat_scheduler,
    channel_checkpoint_path,
    load_channel_gat,
    load_config,
)


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _dataset(config, num_uav):
    local = dict(config, NUM_UAV=int(num_uav))
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 500, "val": 128, "test": 128}
    metadata = expected_dataset_metadata(local, num_uav, seeds, sizes)
    train_val_path, test_path = paper_dataset_paths(
        Path("datasets/paper"), num_uav, seeds, sizes
    )
    train_val = load_validated_dataset(
        train_val_path, metadata, ("train", "val")
    )
    test = load_validated_dataset(test_path, metadata, ("test",))
    return local, train_val, test, metadata


def _train(
    config, num_uav, train_val, test, epochs, train_limit,
    dataset_fingerprint,
):
    _seed(10086)
    model = build_gat_scheduler(config, raw_node_dim=1)
    trainer = Trainer(
        model,
        torch.optim.Adam(model.parameters(), lr=float(config["LR"])),
        device="cpu",
        use_queue=False,
        log_dir=f"logs/paper/scaling/N{num_uav}",
    )
    trained = trainer.train(
        epochs,
        list(train_val["train_set"][:train_limit]),
        train_val["val_set"],
        eval_interval=1,
        use_queue=False,
    )
    audit = _output_audit(model, test["test_set"])
    if not audit["integrity_passed"]:
        raise RuntimeError(f"N={num_uav} Channel-GAT failed integrity audit: {audit}")
    checkpoint = channel_checkpoint_path(config, num_uav)
    if train_limit != 500 or epochs < 30:
        checkpoint = (
            checkpoint.parent / "development"
            / f"quick_N{num_uav}_n{train_limit}_epochs{epochs}.pth"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_version": "channel_gat_v1",
        "model_state_dict": model.state_dict(),
        "tx_power_dbm": 10.0,
        "csi_mode": "perfect",
        "channel_model": "rician",
        "rician_k_db": 5.0,
        "num_uav": int(num_uav),
        "dataset_fingerprint": dataset_fingerprint,
        "config": {
            "NUM_UAV": int(num_uav),
            "TX_POWER": 10.0,
            "CHANNEL_MODEL": "rician",
            "RICIAN_K_DB": 5.0,
            "CSI_MODE": "perfect",
            "TRAIN_SIZE": int(train_limit),
            "VAL_SIZE": 128,
            "TEST_SIZE": 128,
            "SEED": 10086,
        },
        "best_epoch": trained["best_epoch"],
        "best_val_utility": trained["best_val_util"],
        "output_audit": audit,
        "integrity_status": "passed",
        "timestamp": datetime.now().isoformat(),
    }, checkpoint)
    return model, str(checkpoint), trained


def _evaluate(model, dataset):
    trainer = Trainer(
        model, torch.optim.Adam(model.parameters(), lr=1e-4),
        device="cpu", use_queue=False, log_dir="logs/paper/scaling/eval"
    )
    return float(trainer.evaluate(model, dataset, use_queue=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paired", "mixed", "all"), default="all")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20, 30, 40, 50])
    parser.add_argument("--test-sizes", type=int, nargs="+", default=[20, 30, 40, 50])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--use-existing-models", action="store_true")
    parser.add_argument(
        "--output", default="results/paper/channel_gat_scaling/generalization.json"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    epochs = 2 if args.quick else args.epochs
    train_limit = 50 if args.quick else 500
    datasets = {
        n: _dataset(config, n) for n in sorted(set(args.train_sizes + args.test_sizes))
    }
    models, training = {}, {}
    for n in args.train_sizes:
        local, train_val, test, metadata = datasets[n]
        if args.use_existing_models:
            model, checkpoint = load_channel_gat(local, n, device="cpu")
            trained = None
        else:
            model, checkpoint, trained = _train(
                local, n, train_val, test, epochs, train_limit,
                metadata["fingerprint"],
            )
        models[n] = model
        training[n] = {
            "checkpoint": checkpoint,
            "dataset_fingerprint": metadata["fingerprint"],
            "best_epoch": None if trained is None else trained["best_epoch"],
            "best_val_utility": None if trained is None else trained["best_val_util"],
        }

    matrix = {}
    for train_n, model in models.items():
        matrix[str(train_n)] = {}
        for test_n in args.test_sizes:
            if args.mode == "paired" and train_n != test_n:
                continue
            value = _evaluate(model, datasets[test_n][2]["test_set"])
            matrix[str(train_n)][str(test_n)] = value
            print(f"train N={train_n} -> test N={test_n}: {value:.4f}")

    dedicated = {
        str(test_n): matrix[str(test_n)][str(test_n)]
        for test_n in args.test_sizes if str(test_n) in matrix
    }
    relative = {
        train_n: {
            test_n: value / dedicated[test_n]
            for test_n, value in row.items() if test_n in dedicated
        }
        for train_n, row in matrix.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "experiment": "paper_channel_gat_cross_scale_v1",
        "quick": args.quick,
        "reused_existing_models": args.use_existing_models,
        "training": training,
        "absolute_test_utility": matrix,
        "relative_to_test_scale_dedicated_model": relative,
    }, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
