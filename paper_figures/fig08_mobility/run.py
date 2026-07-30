#!/usr/bin/env python3
"""Generate the two Fig. 8 GAT-MP mobile-CSI backlog experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEEDS_MPS = "0,5,10,20"
BLOCK_LENGTHS = "1,5,10,20"
FIXED_SPEED_MPS = "10"
FIXED_BLOCK_LENGTH = "5"


def call(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_backlog_sweep(
    *,
    num_uav: int,
    sweep_name: str,
    speeds_mps: str,
    slots_per_block: str,
    args: argparse.Namespace,
) -> None:
    output = HERE / "data/raw" / f"gatmp_{sweep_name}_N{num_uav}.json"
    call([
        sys.executable,
        "-u",
        "experiments/evaluate_mobile_block_fading_csi.py",
        "--num-uav",
        str(num_uav),
        "--num-slots",
        str(args.slots),
        "--seeds",
        args.seeds,
        "--speeds-mps",
        speeds_mps,
        "--slots-per-block",
        slots_per_block,
        "--stale-blocks",
        "1",
        "--arrival-rates",
        "2.4",
        "--csi-methods",
        "latest,stale",
        "--scheduler-variants",
        "gat_assisted",
        "--rounds",
        "4",
        "--device",
        args.device,
        "--output",
        str(output),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    (HERE / "data/raw").mkdir(parents=True, exist_ok=True)
    for num_uav in [int(value) for value in args.nodes.split(",") if value.strip()]:
        run_backlog_sweep(
            num_uav=num_uav,
            sweep_name="speed",
            speeds_mps=SPEEDS_MPS,
            slots_per_block=FIXED_BLOCK_LENGTH,
            args=args,
        )
        run_backlog_sweep(
            num_uav=num_uav,
            sweep_name="block",
            speeds_mps=FIXED_SPEED_MPS,
            slots_per_block=BLOCK_LENGTHS,
            args=args,
        )


if __name__ == "__main__":
    main()
