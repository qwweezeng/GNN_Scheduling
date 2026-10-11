#!/usr/bin/env python3
"""Train/evaluate the Fig. 4 data-efficiency grid with dual-checkpoint evaluation.

Each (N, train_size, seed) training run:
  1. Trains for 50 epochs with 300-sample canonical pool (first train_size used).
  2. Saves both the best-validation checkpoint (best_model.pth) and the
     final-epoch checkpoint (final_model.pth).
  3. Evaluates both checkpoints on the canonical 128-sample test set using
     three decision rules:
       - soft:       pi used directly without binarization
       - hard:       pi >= 0.5, no feasibility projection (diagnostic only)
       - projected:  pi >= 0.5 + endpoint mutual matching (hard feasible)

Training outputs (per seed, per train_size):
  training.json   — training log + best-epoch info
  best_model.pth — best-validation model state dict
  final_model.pth — final-epoch model state dict (before best restoration)

Evaluation outputs:
  eval_best_soft.json       — soft metric for best checkpoint
  eval_best_hard.json       — hard unprojected metric for best checkpoint
  eval_best_projected.json  — projected metric for best checkpoint
  eval_final_soft.json      — soft metric for final checkpoint
  eval_final_hard.json      — hard unprojected metric for final checkpoint
  eval_final_projected.json — projected metric for final checkpoint

For train_size=300: the centralized full_300 checkpoint pair from the
corresponding Fig. 2 (N, seed) run is referenced; no weight file is copied
and no new training is performed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.project_setup import proposed_gat_checkpoint_dir

HERE = Path(__file__).resolve().parent

SIZES = (10, 20, 30, 40, 60, 80, 100, 200, 300, 500)


def call(cmd: list[str], env: dict | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument(
        "--sizes", default=",".join(str(s) for s in SIZES),
        help=f"Comma-separated training sample sizes; default is {SIZES}",
    )
    parser.add_argument(
        "--train_size", type=int, default=None,
        help="Run a single train_size (overrides --sizes). Use this to run n=500 alone.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--stage",
        choices=("train", "evaluate", "all"),
        default="all",
        help="train=checkpoint/log only; evaluate=existing checkpoints only; all=both.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--fig02-source", default="../fig02_training_curves/data/raw",
        help="Path (relative to HERE) to Fig.2 raw data directory.",
    )
    args = parser.parse_args()

    nodes = [int(x) for x in args.nodes.split(",") if x]
    sizes = [args.train_size] if args.train_size is not None else [int(x) for x in args.sizes.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    fig02_root = (HERE / args.fig02_source).resolve()

    for num_uav in nodes:
        for train_size in sizes:
            for model_seed in seeds:
                artifact_dir = (
                    HERE / "data/raw" / f"N{num_uav}" / f"n{train_size}" / f"seed{model_seed}"
                )
                training_json = artifact_dir / "training.json"
                checkpoint_scope = "full_300" if train_size == 300 else f"prefix_n{train_size}"
                # n=500 uses the same pool as n=300 but trains on the full set
                is_full_500 = train_size == 500
                if is_full_500:
                    checkpoint_scope = "prefix_n500"
                checkpoint_dir = proposed_gat_checkpoint_dir(
                    num_uav, model_seed, checkpoint_scope
                )
                best_checkpoint = checkpoint_dir / "best_model.pth"
                final_checkpoint = checkpoint_dir / "final_model.pth"

                # ── Training ───────────────────────────────────────────────
                if train_size == 300:
                    # Reuse the single canonical full_300 checkpoint pair from Fig. 2.
                    fig02_artifact = fig02_root / f"N{num_uav}" / f"seed{model_seed}"
                    fig02_json = fig02_artifact / "training.json"
                    if not (best_checkpoint.is_file() and final_checkpoint.is_file()):
                        raise FileNotFoundError(
                            f"train_size=300 requires centralized Fig.2 checkpoints at "
                            f"{checkpoint_dir}. Run Fig.2 first."
                        )
                    if not training_json.exists() or args.force:
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        if not fig02_json.is_file():
                            raise FileNotFoundError(
                                f"train_size=300 requires Fig.2 training log at {fig02_json}."
                            )
                        source_report = json.loads(fig02_json.read_text(encoding="utf-8"))
                        training_json.write_text(json.dumps({
                            "artifact_type": "fig04_n300_checkpoint_reference",
                            "checkpoint_scope": "full_300",
                            "source_figure": "fig02_training_curves",
                            "source_training_json": str(fig02_json),
                            "best_checkpoint": str(best_checkpoint),
                            "final_checkpoint": str(final_checkpoint),
                            "best_epoch": source_report["best_epoch"],
                            "best_val_utility": source_report["best_val_utility"],
                            "per_epoch_logs": source_report["per_epoch_logs"],
                        }, indent=2), encoding="utf-8")
                    print(f"reuse Fig.2 checkpoint pair from {checkpoint_dir}", flush=True)
                else:
                    # New training with dual checkpoint output
                    if (
                        not args.force
                        and training_json.exists()
                        and best_checkpoint.exists()
                        and final_checkpoint.exists()
                    ):
                        print(f"reuse training {training_json}", flush=True)
                    else:
                        if args.stage == "evaluate":
                            raise FileNotFoundError(
                                "Fig. 4 evaluation requires training.json and both "
                                f"checkpoints for N={num_uav}, n={train_size}, "
                                f"seed={model_seed}."
                            )
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            sys.executable, "-u",
                            "train/train_channel_gat_rician.py",
                            "--num_uav", str(num_uav),
                            "--train_size", str(train_size),
                            "--prefix_pool_size", "500",
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

                if args.stage == "train":
                    continue

                # ── Evaluation: best checkpoint ─────────────────────────────
                _evaluate("best", best_checkpoint, artifact_dir, num_uav, args, args.gpu_id, train_size)

                # ── Evaluation: final checkpoint ────────────────────────────
                _evaluate("final", final_checkpoint, artifact_dir, num_uav, args, args.gpu_id, train_size)


def _evaluate(
    selection: str,          # "best" or "final"
    checkpoint: Path,
    artifact_dir: Path,
    num_uav: int,
    args: argparse.Namespace,
    gpu_id: int,
    train_size: int,
) -> None:
    """Evaluate a single checkpoint with all three decision rules."""
    soft_out = artifact_dir / f"eval_{selection}_soft.json"
    hard_out = artifact_dir / f"eval_{selection}_hard.json"
    proj_out = artifact_dir / f"eval_{selection}_projected.json"

    if not soft_out.exists() or args.force:
        call([
            sys.executable, "-u",
            "paper_figures/common/evaluate_channel_checkpoint_soft.py",
            "--num-uav", str(num_uav),
            "--train_size", str(train_size),
            "--checkpoint", str(checkpoint),
            "--device", args.device,
            "--gpu-id", str(gpu_id),
            "--output", str(soft_out),
        ])

    if not hard_out.exists() or args.force:
        call([
            sys.executable, "-u",
            "paper_figures/common/evaluate_channel_checkpoint_hard.py",
            "--num-uav", str(num_uav),
            "--train_size", str(train_size),
            "--checkpoint", str(checkpoint),
            "--device", args.device,
            "--gpu-id", str(gpu_id),
            "--output", str(hard_out),
        ])

    if not proj_out.exists() or args.force:
        call([
            sys.executable, "-u",
            "paper_figures/common/evaluate_channel_checkpoint.py",
            "--num-uav", str(num_uav),
            "--train_size", str(train_size),
            "--checkpoint", str(checkpoint),
            "--device", args.device,
            "--gpu-id", str(gpu_id),
            "--output", str(proj_out),
        ])


if __name__ == "__main__":
    main()
