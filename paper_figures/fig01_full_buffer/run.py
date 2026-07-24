#!/usr/bin/env python3
"""Run only the Fig. 1 experiment; plotting is deliberately separate.

Use ``--stage data`` to create/check data, ``--stage train`` to train the
checkpoint, and ``--stage evaluate`` to write benchmark JSON.  Plotting is
intentionally kept in ``plot.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = "Random,Top-SNR,LocalMax,Local Greedy,Spatial DL,FPLinQ,Proposed-GAT"


def call(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--stage", choices=("data", "train", "evaluate", "all"),
        default="evaluate",
        help="data=only dataset; train=only checkpoint; evaluate=only results.",
    )
    parser.add_argument("--prepare-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--train-missing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="replace raw JSON only")
    args = parser.parse_args()
    nodes = [int(x) for x in args.nodes.split(",") if x.strip()]
    stage = args.stage
    # Backward-compatible aliases for this same four-stage workflow.
    if args.prepare_data:
        stage = "data"
    if args.train_missing:
        stage = "train"
    if stage in {"data", "all"}:
        call([sys.executable, "-u", "scripts/prepare_paper_datasets.py", "--num-uavs", args.nodes])
    for n in nodes:
        output = HERE / "data/raw" / f"benchmark_N{n}.json"
        if stage == "data":
            continue
        if stage in {"train", "all"}:
            call([sys.executable, "-u", "train/train_channel_gat_rician.py", "--num_uav", str(n), "--epochs", str(args.epochs)])
        if stage == "train":
            continue
        if output.exists() and not args.force:
            print(f"reuse {output}; pass --force to recompute", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        call([
            sys.executable, "-u", "experiments/compare_gat_vs_spatial_dl.py",
            "--num_uav", str(n), "--num_samples", str(args.samples),
            "--methods", METHODS, "--top_snr_ratio", "0.03", "--output", str(output),
        ])


if __name__ == "__main__":
    main()
