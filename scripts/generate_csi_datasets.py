#!/usr/bin/env python3
"""
Standalone dataset generator for the quantized-CSI channel GAT pipeline.
Generates train / val / test splits WITHOUT any training.
Used by the Fig. 7 Chapter IV batch driver.

Usage:
    uv run python scripts/generate_csi_datasets.py --num-uav 20 --csi-mode quantized --q-bits 4
    uv run python scripts/generate_csi_datasets.py --num-uav 30 --csi-mode large_scale
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.artifacts import (
    atomic_pickle_dump,
    expected_dataset_metadata_quantized,
    paper_dataset_paths_quantized,
    write_dataset_manifest,
)
from utils.io import generate_dataset
from utils.project_setup import build_network_generator, load_config


TRAIN_SIZE = 300
VAL_SIZE   = 128
TEST_SIZE  = 128
SEED       = 10086
VAL_SEED   = 10087
TEST_SEED  = 10088


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--num-uav", type=int, required=True)
    p.add_argument(
        "--csi-mode", default="quantized",
        choices=("perfect", "large_scale", "quantized"),
    )
    p.add_argument(
        "--q-bits", type=int, default=None,
        help="Required when --csi-mode is 'quantized'.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing cache files.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print paths and exit without writing.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    mode   = args.csi_mode
    q_bits = args.q_bits
    n_uav  = args.num_uav

    if mode == "quantized" and q_bits is None:
        raise ValueError("--q-bits is required when --csi-mode is 'quantized'")
    if mode != "quantized" and q_bits is not None:
        raise ValueError("--q-bits may only be set for quantized mode")

    config = load_config(args.config)
    config.update({
        "NUM_UAV": n_uav,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": mode,
        "Q_BITS": q_bits,
    })

    seeds = {"train": SEED, "val": VAL_SEED, "test": TEST_SEED}
    sizes = {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE}
    metadata = expected_dataset_metadata_quantized(
        config, n_uav, seeds, sizes, mode, q_bits,
    )
    train_val_path, test_path = paper_dataset_paths_quantized(
        Path("datasets/paper"), n_uav, seeds, sizes, mode, q_bits,
    )

    print(f"[generate_csi_datasets]")
    print(f"  CSI mode   : {mode}")
    print(f"  Q bits     : {q_bits}")
    print(f"  N UAVs     : {n_uav}")
    print(f"  Train path : {train_val_path}")
    print(f"  Test path  : {test_path}")
    print(f"  Metadata   : {metadata['fingerprint'][:16]}...")

    if args.dry_run:
        return

    if (
        not args.force
        and train_val_path.exists()
        and test_path.exists()
    ):
        print("  -> cache hit, skipping (use --force to overwrite)")
        return

    generator = build_network_generator(config)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"  Generating train ({TRAIN_SIZE})...", flush=True)
    train_set = generate_dataset(
        TRAIN_SIZE, generator,
        include_queue_feature=False,
        csi_mode=mode, q_bits=q_bits,
    )

    random.seed(VAL_SEED)
    np.random.seed(VAL_SEED)
    torch.manual_seed(VAL_SEED)
    print(f"  Generating val ({VAL_SIZE})...", flush=True)
    val_set = generate_dataset(
        VAL_SIZE, generator,
        include_queue_feature=False,
        csi_mode=mode, q_bits=q_bits,
    )

    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    print(f"  Generating test ({TEST_SIZE})...", flush=True)
    test_set = generate_dataset(
        TEST_SIZE, generator,
        include_queue_feature=False,
        csi_mode=mode, q_bits=q_bits,
    )

    atomic_pickle_dump(
        {"metadata": metadata, "train_set": train_set, "val_set": val_set},
        train_val_path,
    )
    atomic_pickle_dump({"metadata": metadata, "test_set": test_set}, test_path)
    write_dataset_manifest(
        train_val_path.parent, metadata, train_val_path, test_path,
    )
    print("  -> done.")


if __name__ == "__main__":
    main()
