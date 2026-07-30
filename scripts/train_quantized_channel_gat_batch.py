#!/usr/bin/env python3
"""
Batch train and evaluate channel GAT checkpoints for every (N, CSI) variant
used in Fig. 7 Chapter IV.

Each variant trains on its own observed-CSI dataset (300/128/128) so the model
learns to make scheduling decisions under the same CSI quality it will face at
test time.  The perfect-CSI checkpoint already exists and is reused as-is.

Usage:
    # Dry-run (print commands without executing)
    uv run python scripts/train_quantized_channel_gat_batch.py --dry-run

    # Full run: data + train + eval for all N=20,30
    uv run python scripts/train_quantized_channel_gat_batch.py --device cuda

    # Stages separately
    uv run python scripts/train_quantized_channel_gat_batch.py --stages data,eval   # just regenerate + eval
    uv run python scripts/train_quantized_channel_gat_batch.py --stages train,eval   # train + eval

    # Override defaults
    uv run python scripts/train_quantized_channel_gat_batch.py \
        --num-uavs 20 30 40 --epochs 50 --device cuda --dry-run
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "train" / "train_channel_gat_rician.py"
EVAL_SCRIPT  = ROOT / "experiments" / "evaluate_channel_gat_csi.py"
DATA_SCRIPT  = ROOT / "scripts" / "generate_csi_datasets.py"
OUT_DIR      = ROOT / "paper_figures" / "fig07_csi_feedback" / "data" / "raw"

ALL_NUM_UAVS  = [20, 30]
CSI_VARIANTS  = [
    ("large_scale", None),
    ("quantized",   2),
    ("quantized",   4),
    ("quantized",   6),
    ("quantized",   8),
    ("quantized",  10),
]
PERFECT = ("perfect", None)

TRAIN_DEFAULTS = dict(
    train_size=300, val_size=128, test_size=128,
    epochs=50, learning_rate=1e-4,
    seed=10086, val_seed=10087, test_seed=10088,
    rician_k_db=5.0,
)


def csi_tag(mode, q_bits):
    if mode == "perfect":     return "perfect"
    if mode == "large_scale": return "large_scale"
    return f"quantized_{int(q_bits)}bit"


def _base_cmd(n_uav, mode, q_bits, epochs, device):
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", "configs/paper.yaml",
        "--num_uav", str(n_uav),
        "--epochs", str(epochs),
        "--seed", str(TRAIN_DEFAULTS["seed"]),
        "--val_seed", str(TRAIN_DEFAULTS["val_seed"]),
        "--test_seed", str(TRAIN_DEFAULTS["test_seed"]),
        "--rician_k_db", str(TRAIN_DEFAULTS["rician_k_db"]),
        "--learning_rate", str(TRAIN_DEFAULTS["learning_rate"]),
        "--device", device,
        "--csi-mode", mode,
    ]
    if q_bits is not None:
        cmd += ["--q-bits", str(q_bits)]
    return cmd


def dataset_cmd(n_uav, mode, q_bits):
    cmd = [
        sys.executable, str(DATA_SCRIPT),
        "--num-uav", str(n_uav),
        "--csi-mode", mode,
    ]
    if q_bits is not None:
        cmd += ["--q-bits", str(q_bits)]
    return cmd


def train_cmd(n_uav, mode, q_bits, epochs, device):
    return _base_cmd(n_uav, mode, q_bits, epochs, device)


def eval_cmd(n_uav, mode, q_bits, device):
    tag = csi_tag(mode, q_bits)
    output = OUT_DIR / f"chapter4_N{n_uav}_v2_{tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--num-uav", str(n_uav),
        "--device", device,
        "--csi-mode", mode,
        "--output", str(output),
    ]
    if q_bits is not None:
        cmd += ["--q-bits", str(q_bits)]
    return cmd, output, tag


def run_cmd(cmd, label, dry_run):
    print(f"\n{'='*60}\n  [{label}]\n{'='*60}\n$ {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {label} (exit {result.returncode})")
    return ok


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stages", default="data,train,eval",
        help="Comma-separated: data,train,eval (default: all)",
    )
    p.add_argument(
        "--num-uavs", type=int, nargs="+", default=ALL_NUM_UAVS,
    )
    p.add_argument(
        "--epochs", type=int, default=TRAIN_DEFAULTS["epochs"],
    )
    p.add_argument(
        "--device", default="cuda", choices=("cpu", "cuda"),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force-regenerate", action="store_true",
        help="Force dataset re-generation (skip cache check)",
    )
    p.add_argument(
        "--perfect-csi", action="store_true",
        help="Regenerate the perfect-CSI checkpoint too (default: reuse existing)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    stages = set(args.stages.split(","))
    device = args.device
    print(f"Stages    : {stages}")
    print(f"N         : {args.num_uavs}")
    print(f"Epochs    : {args.epochs}")
    print(f"Device    : {device}")
    print(f"Dry-run   : {args.dry_run}")
    print(f"Force-regen: {args.force_regenerate}")

    # ---- assemble all jobs ----
    jobs = []  # (n_uav, mode, q_bits, needs_train)
    for n_uav in args.num_uavs:
        # perfect: reuse existing checkpoint unless --perfect-csi
        jobs.append((n_uav, PERFECT[0], PERFECT[1], bool(args.perfect_csi)))
        for mode, q_bits in CSI_VARIANTS:
            jobs.append((n_uav, mode, q_bits, True))

    failed = []
    for n_uav, mode, q_bits, needs_train in jobs:
        tag = csi_tag(mode, q_bits)
        job_label = f"{tag} N={n_uav}"

        # ---- data ----
        if "data" in stages:
            from utils.artifacts import paper_dataset_paths_quantized
            root = Path("datasets/paper")
            seeds = {"train": TRAIN_DEFAULTS["seed"],
                     "val":  TRAIN_DEFAULTS["val_seed"],
                     "test": TRAIN_DEFAULTS["test_seed"]}
            sizes = {"train": TRAIN_DEFAULTS["train_size"],
                     "val":  TRAIN_DEFAULTS["val_size"],
                     "test": TRAIN_DEFAULTS["test_size"]}
            tv_p, test_p = paper_dataset_paths_quantized(
                root, n_uav, seeds, sizes, mode, q_bits)

            if not args.force_regenerate and tv_p.exists() and test_p.exists():
                print(f"[SKIP data] {job_label}: cache hit")
            else:
                cmd = dataset_cmd(n_uav, mode, q_bits)
                if args.force_regenerate:
                    cmd.append("--force")
                ok = run_cmd(cmd, f"data {job_label}", args.dry_run)
                if not ok:
                    failed.append(f"data {job_label}")

        # ---- train ----
        if "train" in stages and needs_train:
            cmd = train_cmd(n_uav, mode, q_bits, args.epochs, device)
            ok = run_cmd(cmd, f"train {job_label}", args.dry_run)
            if not ok:
                failed.append(f"train {job_label}")

        # ---- eval ----
        if "eval" in stages:
            cmd, out_path, _ = eval_cmd(n_uav, mode, q_bits, device)
            if not args.force_regenerate and out_path.exists():
                print(f"[SKIP eval] {job_label}: {out_path} exists")
            else:
                ok = run_cmd(cmd, f"eval {job_label}", args.dry_run)
                if not ok:
                    failed.append(f"eval {job_label}")

    print()
    if failed:
        print(f"FAILED: {failed}")
    else:
        print("All stages completed successfully.")


if __name__ == "__main__":
    main()
