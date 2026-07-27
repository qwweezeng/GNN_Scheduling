#!/usr/bin/env python3
"""Run only the Fig. 1 experiment; plotting is deliberately separate.

Use ``--stage data`` to create/check data, ``--stage train`` to train the
checkpoint, and ``--stage evaluate`` to write benchmark JSON.  Plotting is
intentionally kept in ``plot.py``.

Fig. 1 reuses the corresponding Fig. 2 ``full_300`` checkpoint for
``seed=0``.  Therefore ``--stage train`` only verifies that the shared
checkpoint exists; it never starts another training job.  The best-validation
checkpoint is used automatically in the ``--stage evaluate`` phase.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.project_setup import proposed_gat_checkpoint_dir

HERE = Path(__file__).resolve().parent
METHODS = "Random,Top-SNR,Local Greedy,Spatial DL,FPLinQ,Proposed-GAT"


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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--force",
        action="store_true",
        help="back up and then recompute each existing raw JSON",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--methods", default=METHODS,
        help="Comma-separated methods to evaluate; defaults to the paper set.",
    )
    parser.add_argument(
        "--gat-model",
        default="",
        help=(
            "Path to a channel-GAT checkpoint.  When empty the script "
            "searches the production checkpoint path.  When non-empty the "
            "supplied checkpoint is used for the Proposed-GAT method."
        ),
    )
    args = parser.parse_args()
    nodes = [int(x) for x in args.nodes.split(",") if x.strip()]

    if args.stage in {"data", "all"}:
        call([sys.executable, "-u", "scripts/prepare_paper_datasets.py", "--num-uavs", args.nodes])

    for n in nodes:
        suffix = ""
        if args.methods != METHODS:
            suffix += "_gat_only"
        output = HERE / "data/raw" / f"benchmark_N{n}{suffix}.json"

        if args.stage == "data":
            continue

        checkpoint_dir = proposed_gat_checkpoint_dir(n, 0, "full_300")
        best_checkpoint = checkpoint_dir / "best_model.pth"
        final_checkpoint = checkpoint_dir / "final_model.pth"
        gat_model = args.gat_model
        if args.stage in {"train", "all"}:
            if not best_checkpoint.is_file() or not final_checkpoint.is_file():
                raise FileNotFoundError(
                    "Fig. 1 reuses Fig. 2 full_300 checkpoints, but the pair is "
                    f"missing from {checkpoint_dir}. Run the Fig. 2 phase first."
                )
            print(f"reuse Fig. 2 checkpoint pair: {checkpoint_dir}", flush=True)
            gat_model = str(best_checkpoint)

        if args.stage == "evaluate" and not gat_model and best_checkpoint.is_file():
            gat_model = str(best_checkpoint)

        if args.stage == "train":
            continue

        # Use the trained checkpoint if available, otherwise fall back to user's model
        effective_gat_model = gat_model or args.gat_model

        if output.exists() and not args.force:
            print(f"reuse {output}; pass --force to recompute", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            backup = output.with_name(f"{output.stem}.before_rerun_{timestamp}.json")
            shutil.copy2(output, backup)
            print(f"backup {output} -> {backup}", flush=True)

        cmd = [
            sys.executable, "-u",
            "experiments/compare_gat_vs_spatial_dl.py",
            "--num_uav", str(n),
            "--num_samples", str(args.samples),
            "--methods", args.methods,
            "--top_snr_ratio", "0.02",
            "--device", args.device,
            "--output", str(output),
        ]
        if effective_gat_model:
            cmd.extend(["--gat_model", effective_gat_model])
        call(cmd)


if __name__ == "__main__":
    main()
