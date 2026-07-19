#!/usr/bin/env python3
"""Paper Fig. 2: Channel-GAT convergence and training-set-size study.

All points use prefixes of one canonical 500-sample training bank and share
the same 128-sample validation and test banks.
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

from train.trainer import Trainer  # noqa: E402
from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.project_setup import build_gat_scheduler, load_config  # noqa: E402


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument(
        "--train_sizes", type=int, nargs="+", default=[50, 100, 200, 300, 500]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument(
        "--output", default="results/paper/channel_gat_training/sample_efficiency_N20.json"
    )
    args = parser.parse_args()

    if min(args.train_sizes) <= 0 or max(args.train_sizes) > 500:
        raise ValueError("train_sizes must be between 1 and the canonical pool size 500")
    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 500, "val": 128, "test": 128}
    expected = expected_dataset_metadata(config, args.num_uav, seeds, sizes)
    train_val_path, test_path = paper_dataset_paths(
        Path("datasets/paper"), args.num_uav, seeds, sizes
    )
    train_val = load_validated_dataset(
        train_val_path, expected, ("train", "val")
    )
    test = load_validated_dataset(test_path, expected, ("test",))

    runs = []
    for train_size in args.train_sizes:
        _seed(args.seed)
        model = build_gat_scheduler(config, raw_node_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["LR"]))
        trainer = Trainer(
            model,
            optimizer,
            device="cpu",
            use_queue=False,
            log_dir=f"logs/paper/convergence/N{args.num_uav}_n{train_size}",
        )
        trained = trainer.train(
            num_epochs=args.epochs,
            train_dataset=list(train_val["train_set"][:train_size]),
            val_dataset=train_val["val_set"],
            eval_interval=1,
            use_queue=False,
        )
        test_utility = float(
            trainer.evaluate(model, test["test_set"], use_queue=False)
        )
        run = {
            "train_size": train_size,
            "best_epoch": trained["best_epoch"],
            "best_val_utility": trained["best_val_util"],
            "test_utility": test_utility,
            "per_epoch": trained["per_epoch_logs"],
        }
        runs.append(run)
        print(
            f"train_size={train_size}: test={test_utility:.4f}, "
            f"best_epoch={trained['best_epoch']}"
        )

    output = Path(args.output)
    if output.name.endswith("N20.json") and args.num_uav != 20:
        output = output.with_name(f"sample_efficiency_N{args.num_uav}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "experiment": "paper_channel_gat_sample_efficiency_v1",
        "num_uav": args.num_uav,
        "shared_train_val_path": str(train_val_path),
        "shared_test_path": str(test_path),
        "dataset_fingerprint": expected["fingerprint"],
        "training_seed_reused_for_fair_initialization": args.seed,
        "runs": runs,
    }, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
