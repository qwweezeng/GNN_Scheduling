#!/usr/bin/env python3
"""Generate or strictly validate the static Rician datasets used by the paper.

The default order is largest-first so the slowest N=50 cache is prepared
first.  Existing files are reused only after metadata and split-size checks.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.artifacts import (  # noqa: E402
    atomic_pickle_dump,
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
    write_dataset_manifest,
)
from utils.io import generate_dataset  # noqa: E402
from utils.project_setup import build_network_generator, load_config  # noqa: E402


def csv_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def generate_split(generator, size: int, seed: int, batch_size: int):
    set_seed(seed)
    return generate_dataset(
        int(size),
        generator,
        batch_size=int(batch_size),
        csi_mode="perfect",
        q_bits=4,
        include_queue_feature=False,
    )


def reusable_prefix_path(root: Path, num_uav: int, seeds: dict, sizes: dict) -> Path:
    source_sizes = dict(sizes, train=500)
    source_path, _ = paper_dataset_paths(root, num_uav, seeds, source_sizes)
    return source_path


def load_reusable_prefix(path: Path, size: int, seeds: dict, config: dict, num_uav: int):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    source_metadata = payload.get("metadata", {})
    expected_source = expected_dataset_metadata(
        config, num_uav, seeds,
        {"train": 500, "val": len(payload.get("val_set", ())), "test": 128},
    )
    if source_metadata != expected_source:
        raise ValueError(f"prefix source metadata mismatch: {path}")
    if len(payload.get("train_set", ())) < size:
        raise ValueError(f"prefix source has fewer than {size} training samples: {path}")
    return payload["train_set"][:size], payload["val_set"]


def prepare_one(args, base_config: dict, num_uav: int) -> None:
    config = dict(base_config)
    config.update({
        "NUM_UAV": int(num_uav),
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
        "USE_QUEUE": False,
        "NODE_RAW_DIM": 1,
    })
    seeds = {"train": args.train_seed, "val": args.val_seed, "test": args.test_seed}
    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    metadata = expected_dataset_metadata(config, num_uav, seeds, sizes)
    train_val_path, test_path = paper_dataset_paths(
        Path(args.output_root), num_uav, seeds, sizes
    )

    valid = False
    if train_val_path.exists() and test_path.exists() and not args.force:
        try:
            load_validated_dataset(train_val_path, metadata, ("train", "val"))
            load_validated_dataset(test_path, metadata, ("test",))
            valid = True
        except ValueError as error:
            print(f"[N={num_uav}] cache rejected: {error}", flush=True)
    if valid:
        manifest = write_dataset_manifest(
            train_val_path.parent, metadata, train_val_path, test_path
        )
        print(f"[N={num_uav}] validated existing dataset; manifest={manifest}", flush=True)
        return

    prefix_path = reusable_prefix_path(
        Path(args.output_root), num_uav, seeds, sizes
    )
    if not args.force and prefix_path.exists() and int(args.train_size) <= 500:
        print(f"[N={num_uav}] reusing first {args.train_size} samples from {prefix_path.name}", flush=True)
        train_set, val_set = load_reusable_prefix(
            prefix_path, args.train_size, seeds, config, num_uav
        )
    else:
        generator = build_network_generator(config)
        print(f"[N={num_uav}] generating train={args.train_size}", flush=True)
        train_set = generate_split(generator, args.train_size, args.train_seed, args.batch_size)
        print(f"[N={num_uav}] generating val={args.val_size}", flush=True)
        val_set = generate_split(generator, args.val_size, args.val_seed, args.batch_size)
    atomic_pickle_dump(
        {"metadata": metadata, "train_set": train_set, "val_set": val_set},
        train_val_path,
    )
    del train_set, val_set

    if test_path.exists() and not args.force:
        with test_path.open("rb") as handle:
            existing_test = pickle.load(handle)
        test_set = existing_test.get("test_set")
        if len(test_set or ()) != int(args.test_size):
            test_set = None
    else:
        test_set = None
    if test_set is None:
        generator = build_network_generator(config)
        print(f"[N={num_uav}] generating test={args.test_size}", flush=True)
        test_set = generate_split(generator, args.test_size, args.test_seed, args.batch_size)
    atomic_pickle_dump(
        {"metadata": metadata, "test_set": test_set}, test_path
    )
    manifest = write_dataset_manifest(
        train_val_path.parent, metadata, train_val_path, test_path
    )
    print(f"[N={num_uav}] complete; manifest={manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num-uavs", default="50,40,30,20")
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=10086)
    parser.add_argument("--val-seed", type=int, default=10087)
    parser.add_argument("--test-seed", type=int, default=10088)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output-root", default="datasets/paper")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for num_uav in csv_ints(args.num_uavs):
        prepare_one(args, config, num_uav)


if __name__ == "__main__":
    main()
