#!/usr/bin/env python3
"""Fig. 3 uses the seed-resolved frozen-GAT training reports from Fig. 2.

No additional model is trained: this is deliberately an audit of the same
training trajectories.  Fig. 3 consumes ``training.json`` only; model weights
remain centralized in the checkpoint store and are not duplicated here.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="../fig02_training_curves/data/raw",
        help="Path (relative to HERE) to Fig.2 raw data directory.",
    )
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--force", action="store_true")
    a = parser.parse_args()

    fig02_root = (HERE / a.source).resolve()
    nodes = [int(x) for x in a.nodes.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]

    # Fig. 3 audits epoch logs only. Checkpoints remain in the centralized store.
    missing = []
    for n in nodes:
        for s in seeds:
            report = fig02_root / f"N{n}" / f"seed{s}" / "training.json"
            if not report.is_file():
                missing.append(str(report))
    if missing:
        raise FileNotFoundError(
            "Run Fig.2 first; missing training reports:\n" + "\n".join(missing[:8])
        )

    # Copy only the training trajectories needed by the Fig. 3 audit.
    for n in nodes:
        for s in seeds:
            src = fig02_root / f"N{n}" / f"seed{s}" / "training.json"
            dst_dir = HERE / "data/raw" / f"N{n}" / f"seed{s}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "training.json"
            if not dst.exists() or a.force:
                shutil.copy2(src, dst)

    print(f"Fig.3 audit source verified and synced from: {fig02_root}", flush=True)


if __name__ == "__main__":
    main()
