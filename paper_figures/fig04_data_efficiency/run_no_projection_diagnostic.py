#!/usr/bin/env python3
"""Evaluate all existing Fig. 4 checkpoints without endpoint projection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def call(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--sizes", default="10,20,30,40,50,60,80,100,200,300")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    nodes = [int(value) for value in args.nodes.split(",") if value]
    sizes = [int(value) for value in args.sizes.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    for num_uav in nodes:
        for train_size in sizes:
            for model_seed in seeds:
                artifact_dir = HERE / "data/raw" / f"N{num_uav}" / f"n{train_size}" / f"seed{model_seed}"
                checkpoint = artifact_dir / "model.pth"
                output = artifact_dir / "no_projection_evaluation.json"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing Fig. 4 checkpoint: {checkpoint}")
                if output.exists() and not args.force:
                    continue
                call([
                    sys.executable,
                    "-u",
                    "paper_figures/common/evaluate_channel_checkpoint_no_projection.py",
                    "--num-uav",
                    str(num_uav),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    args.device,
                    "--output",
                    str(output),
                ])


if __name__ == "__main__":
    main()
