#!/usr/bin/env python3
"""Reproducible staged pipeline for TNSE Fig. 7 mobile/aged CSI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = HERE / "data/raw"
RUNNER = ROOT / "experiments/evaluate_mobile_block_fading_csi.py"


def call(args):
    command = [sys.executable, "-u", str(RUNNER), *args]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def common(slots, seeds, output, schedulers="gat_assisted,direct_link", device="cpu"):
    return [
        "--num-uav", "20", "--num-slots", str(slots), "--seeds", seeds,
        "--rounds", "4", "--q-bits", "4", "--scheduler-variants", schedulers,
        "--device", device, "--output", str(output),
    ]


SLICE_FILES = {
    "speed": "speed.json",
    "block_length": "block_length.json",
    "stale_age": "stale_age.json",
    "load": "load.json",
}


def run_formal(
    slots=5000,
    seeds="42,43,44,45,46",
    schedulers="gat_assisted,direct_link",
    device="cpu",
    slices=None,
):
    RAW.mkdir(parents=True, exist_ok=True)
    selected = set(slices or SLICE_FILES)
    if "speed" in selected:
        call(common(slots, seeds, RAW / "speed.json", schedulers, device) + [
            "--speeds-mps", "0,5,10,20", "--slots-per-block", "5",
            "--stale-blocks", "1", "--arrival-rates", "2.4",
            "--csi-methods", "latest,stale,large_scale,quantized_4bit",
        ])
    if "block_length" in selected:
        call(common(slots, seeds, RAW / "block_length.json", schedulers, device) + [
            "--speeds-mps", "10", "--slots-per-block", "1,5,10,20",
            "--stale-blocks", "1", "--arrival-rates", "2.4",
            "--csi-methods", "latest,stale",
        ])
    if "stale_age" in selected:
        call(common(slots, seeds, RAW / "stale_age.json", schedulers, device) + [
            "--speeds-mps", "10", "--slots-per-block", "5",
            "--stale-blocks", "1,2,4,8", "--arrival-rates", "2.4",
            "--csi-methods", "latest,stale",
        ])
    if "load" in selected:
        call(common(slots, seeds, RAW / "load.json", schedulers, device) + [
            "--speeds-mps", "10", "--slots-per-block", "5",
            "--stale-blocks", "1", "--arrival-rates", "2,2.4,2.8,3.2",
            "--csi-methods", "latest,stale",
        ])


def validate(slices=None):
    selected = set(slices or SLICE_FILES)
    expected = {SLICE_FILES[name] for name in selected}
    missing = [name for name in expected if not (RAW / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Fig. 7 raw outputs: {missing}")
    for name in sorted(expected):
        payload = json.loads((RAW / name).read_text(encoding="utf-8"))
        if payload.get("channel_gat_checkpoint") is None:
            raise ValueError(f"{name}: missing audited GAT checkpoint")
        num_slots = int(payload.get("settings", {}).get("num_slots", 0))
        for row in payload.get("records", []):
            if abs(float(row.get("packet_count_conservation_error", 1))) > 1e-6:
                raise ValueError(f"{name}: packet conservation failed")
            # Queue work is accumulated in float32 packet units.  At formal
            # scale its absolute ULP is around 1e-3, so audit a strict relative
            # bound as well as a small absolute floor instead of demanding an
            # unrepresentable 1e-4 absolute residual.
            expected_work = num_slots * float(row.get("arrival_rate", 0.0))
            work_tolerance = max(1e-4, 2e-7 * max(expected_work, 1.0))
            if abs(float(row.get("packet_work_conservation_error", 1))) > work_tolerance:
                raise ValueError(
                    f"{name}: work conservation failed "
                    f"(error={row.get('packet_work_conservation_error')}, "
                    f"tolerance={work_tolerance})"
                )
            if not bool(row.get("shared_uav_feasible_all_slots", False)):
                raise ValueError(f"{name}: shared-UAV feasibility audit failed")


def manifest(command):
    artifacts = {}
    for path in HERE.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            artifacts[str(path.relative_to(HERE))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (HERE / "manifest.json").write_text(json.dumps({
        "schema": "tnse_figure_manifest_v1",
        "figure": "fig07_mobile_csi",
        "created_at": datetime.now().astimezone().isoformat(),
        "frozen_git_revision": "ad2ee67d76e05effa4953db18ab677502ddd3ba1",
        "command": command,
        "artifacts": artifacts,
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "run", "plot", "all", "validate"))
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--include-fplinq", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--slice", default="all",
        choices=("all", *SLICE_FILES),
        help="run one independent formal slice; use four server jobs in parallel",
    )
    args = parser.parse_args()
    schedulers = "gat_assisted,direct_link" + (",weighted_fplinq" if args.include_fplinq else "")
    selected = list(SLICE_FILES) if args.slice == "all" else [args.slice]
    if args.stage == "smoke":
        output = HERE / "data/smoke/speed.json"
        call(common(100, "42,43", output, device=args.device) + [
            "--speeds-mps", "0,10", "--slots-per-block", "5",
            "--stale-blocks", "1", "--arrival-rates", "2.4",
            "--csi-methods", "latest,stale",
        ])
    elif args.stage in ("run", "all"):
        run_formal(args.slots, args.seeds, schedulers, args.device, selected)
        validate(selected)
    elif args.stage == "validate":
        validate(selected)
    if args.stage in ("plot", "all"):
        validate()
        subprocess.run([sys.executable, "-u", str(HERE / "plot.py")], cwd=ROOT, check=True)
    manifest([sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
