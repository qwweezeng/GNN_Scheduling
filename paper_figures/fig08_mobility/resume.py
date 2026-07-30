#!/usr/bin/env python3
"""Resume Fig. 8 mobility experiments without replacing completed outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEEDS_MPS = "0,5,10,20"
BLOCK_LENGTHS = "1,5,10,20"
METHODS = {"latest", "stale"}


def call(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def complete(path: Path, sweep: str) -> bool:
    values = SPEEDS_MPS.split(",") if sweep == "speed" else BLOCK_LENGTHS.split(",")
    expected = {
        (float(value) if sweep == "speed" else int(value), method)
        for value in values
        for method in METHODS
    }
    found = {
        (record["speed_mps"] if sweep == "speed" else record["slots_per_block"], record["method"])
        for record in records(path)
    }
    return expected.issubset(found)


def resume_sweep(num_uav: int, sweep: str, args: argparse.Namespace) -> None:
    output = HERE / "data/raw" / f"gatmp_{sweep}_N{num_uav}.json"
    progress = output.with_suffix(".progress.json")
    if complete(output, sweep) or complete(progress, sweep):
        print(f"[skip] N={num_uav} {sweep} sweep is complete", flush=True)
        return

    speeds = SPEEDS_MPS if sweep == "speed" else "10"
    blocks = "5" if sweep == "speed" else BLOCK_LENGTHS
    call([
        sys.executable, "-u", "experiments/evaluate_mobile_block_fading_csi.py",
        "--num-uav", str(num_uav), "--num-slots", str(args.slots),
        "--seeds", args.seeds, "--speeds-mps", speeds,
        "--slots-per-block", blocks, "--stale-blocks", "1",
        "--arrival-rates", "2.4", "--csi-methods", "latest,stale",
        "--scheduler-variants", "gat_assisted", "--rounds", "4",
        "--device", args.device, "--output", str(output),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    for num_uav in (int(value) for value in args.nodes.split(",") if value.strip()):
        resume_sweep(num_uav, "speed", args)
        resume_sweep(num_uav, "block", args)


if __name__ == "__main__":
    main()
