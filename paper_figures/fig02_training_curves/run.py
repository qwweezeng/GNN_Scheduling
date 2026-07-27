#!/usr/bin/env python3
"""Train the 4 x 3 frozen-architecture replicas needed by Fig. 2/3/4.

Outputs seed-resolved reports with dual checkpoints:
  best_model.pth  — model state dict at best validation utility epoch
  final_model.pth — model state dict at final training epoch (before best restoration)

Both checkpoints are also evaluated on the canonical test set in three modes:
  soft / hard_unprojected / projected.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.project_setup import proposed_gat_checkpoint_dir

HERE = Path(__file__).resolve().parent


def call(cmd: list[str], env: dict | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    nodes = [int(x) for x in args.nodes.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]

    for num_uav in nodes:
        for model_seed in seeds:
            artifact_dir = HERE / "data/raw" / f"N{num_uav}" / f"seed{model_seed}"
            checkpoint_dir = proposed_gat_checkpoint_dir(
                num_uav, model_seed, "full_300"
            )
            training_json = artifact_dir / "training.json"
            best_checkpoint = checkpoint_dir / "best_model.pth"
            final_checkpoint = checkpoint_dir / "final_model.pth"

            if (
                training_json.exists()
                and best_checkpoint.exists()
                and final_checkpoint.exists()
                and not args.force
            ):
                print(f"reuse {training_json}", flush=True)
                _evaluate("best", best_checkpoint, artifact_dir, num_uav, args)
                _evaluate("final", final_checkpoint, artifact_dir, num_uav, args)
                continue

            artifact_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-u",
                "train/train_channel_gat_rician.py",
                "--num_uav", str(num_uav),
                "--train_size", "300",
                "--val_size", "128",
                "--test_size", "128",
                "--epochs", str(args.epochs),
                "--seed", "10086",
                "--val_seed", "10087",
                "--test_seed", "10088",
                "--model-seed", str(model_seed),
                "--checkpoint_out", str(best_checkpoint),
                "--final-checkpoint_out", str(final_checkpoint),
                "--report_out", str(training_json),
                "--device", args.device,
                "--gpu-id", str(args.gpu_id),
            ]
            call(cmd)

            # Evaluate both checkpoints on canonical test set
            _evaluate("best", best_checkpoint, artifact_dir, num_uav, args)
            _evaluate("final", final_checkpoint, artifact_dir, num_uav, args)


def _evaluate(
    selection: str,
    checkpoint: Path,
    artifact_dir: Path,
    num_uav: int,
    args: argparse.Namespace,
) -> None:
    soft_out = artifact_dir / f"eval_{selection}_soft.json"
    hard_out = artifact_dir / f"eval_{selection}_hard.json"
    proj_out = artifact_dir / f"eval_{selection}_projected.json"

    if not soft_out.exists() or args.force:
        call([
            sys.executable, "-u",
            "paper_figures/common/evaluate_channel_checkpoint_soft.py",
            "--num-uav", str(num_uav),
            "--checkpoint", str(checkpoint),
            "--device", args.device,
            "--gpu-id", str(args.gpu_id),
            "--output", str(soft_out),
        ])

    if not hard_out.exists() or args.force:
        call([
            sys.executable, "-u",
            "paper_figures/common/evaluate_channel_checkpoint_hard.py",
            "--num-uav", str(num_uav),
            "--checkpoint", str(checkpoint),
            "--device", args.device,
            "--gpu-id", str(args.gpu_id),
            "--output", str(hard_out),
        ])

    if not proj_out.exists() or args.force:
        call([
            sys.executable, "-u",
            "paper_figures/common/evaluate_channel_checkpoint.py",
            "--num-uav", str(num_uav),
            "--checkpoint", str(checkpoint),
            "--device", args.device,
            "--gpu-id", str(args.gpu_id),
            "--output", str(proj_out),
        ])


if __name__ == "__main__":
    main()
