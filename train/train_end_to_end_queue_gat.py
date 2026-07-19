#!/usr/bin/env python3
"""Train the direct end-to-end GAT-q baseline under the formal physics.

The model sees ``[SNR, Q/max(Q)]`` and directly minimizes the one-slot
queue-weighted rate loss.  It is retained as an intentionally simple baseline,
not as the proposed two-stage method.
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.trainer import Trainer, compute_queue_feature_stats  # noqa: E402
from utils.io import generate_dataset  # noqa: E402
from utils.project_setup import (  # noqa: E402
    build_gat_scheduler,
    build_network_generator,
    queue_checkpoint_path,
    queue_training_cache_path,
    load_config,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_generate(config, cache_path, sizes, queue_range, seed):
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            payload = pickle.load(file)
        metadata = payload.get("metadata", payload.get("artifact_physics", {}))
        expected = {
            "num_uav": int(config["NUM_UAV"]),
            "tx_power_dbm": float(config["TX_POWER"]),
            "csi_mode": str(config["CSI_MODE"]),
            "channel_model": str(config["CHANNEL_MODEL"]).lower(),
            "rician_k_db": float(config["RICIAN_K_DB"]),
            "queue_range": list(queue_range),
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            if all(len(payload[f"{split}_set"]) >= sizes[split] for split in sizes):
                print(f"[Cache hit] {cache_path}")
                return {
                    split: payload[f"{split}_set"][:sizes[split]]
                    for split in sizes
                }
        print(f"[Cache mismatch] regenerating {cache_path}")

    generator = build_network_generator(config)
    datasets = {}
    offsets = {"train": 0, "val": 100_003, "test": 200_003}
    for split, size in sizes.items():
        set_seed(seed + offsets[split])
        print(f"[Generate] {split}: {size} queue-aware snapshots")
        datasets[split] = generate_dataset(
            size,
            generator,
            batch_size=config.get("GEN_BATCH_SIZE", 50),
            csi_mode="perfect",
            include_queue_feature=True,
            queue_mode="random",
            queue_range=queue_range,
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as file:
        pickle.dump({
            "format": "formal_queue_training_v1",
            "metadata": {
                "num_uav": int(config["NUM_UAV"]),
                "tx_power_dbm": float(config["TX_POWER"]),
                "csi_mode": str(config["CSI_MODE"]),
                "channel_model": str(config["CHANNEL_MODEL"]).lower(),
                "rician_k_db": float(config["RICIAN_K_DB"]),
                "queue_range": list(queue_range),
                "seed": int(seed),
            },
            **{f"{split}_set": data for split, data in datasets.items()},
        }, file)
    print(f"[Cache saved] {cache_path}")
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--queue_max", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CSI_MODE": "perfect",
        "SEED": args.seed,
    })
    queue_range = (0, args.queue_max)
    cache_path = queue_training_cache_path(config, args.num_uav)
    checkpoint_path = queue_checkpoint_path(config, args.num_uav)
    sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    datasets = load_or_generate(
        config, cache_path, sizes, queue_range, args.seed
    )

    mean, std = compute_queue_feature_stats(
        datasets["train"], queue_range=queue_range, seed=args.seed
    )
    model = build_gat_scheduler(
        config,
        raw_node_dim=2,
        input_feature_mean=mean,
        input_feature_std=std,
        use_input_skip=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        log_dir=config.get("LOG_DIR", "./logs"),
        use_queue=True,
        bandwidth=config.get("BANDWIDTH", 5e6),
        slot_duration=config.get("SLOT_DURATION", 0.001),
        packet_size=config.get("PACKET_SIZE", 12000),
        queue_range=queue_range,
        queue_seed=args.seed,
    )
    trainer.train(
        num_epochs=args.epochs,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        eval_interval=config.get("EVAL_INTERVAL", 1),
        lr_scheduler=lr_scheduler,
        use_queue=True,
    )
    test_utility = trainer.evaluate(
        model, datasets["test"], use_queue=True
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_version": "mws_v3",
        "model_state_dict": model.state_dict(),
        "input_feature_mean": model.input_feature_mean.detach().cpu(),
        "input_feature_std": model.input_feature_std.detach().cpu(),
        "queue_range": list(queue_range),
        "queue_seed": args.seed,
        "queue_feature": "relative_max",
        "training_decision": "stochastic_ste",
        "stochastic_ste": True,
        "decision_threshold": float(model.decision_threshold),
        "initial_schedule_bias": 0.0,
        "use_input_skip": True,
        "tx_power_dbm": 10.0,
        "csi_mode": "perfect",
        "channel_model": str(config["CHANNEL_MODEL"]).lower(),
        "rician_k_db": float(config["RICIAN_K_DB"]),
        "num_uav": int(args.num_uav),
        "training_sizes": sizes,
        "epochs": args.epochs,
        "test_hard_mws_utility": float(test_utility),
    }, checkpoint_path)
    print(f"[Saved] {checkpoint_path}")
    print(f"[Test hard MWS utility] {test_utility:.6f}")


if __name__ == "__main__":
    main()
