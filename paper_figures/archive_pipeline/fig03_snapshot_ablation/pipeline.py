#!/usr/bin/env python3
"""Staged, reproducible pipeline for TNSE Fig. 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
RAW_DIR = FIGURE_DIR / "data/raw"
SMOKE_DIR = FIGURE_DIR / "data/smoke"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-scheduling-mpl")


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _validate(path: Path, num_graphs: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    expected = int(num_graphs) * 3 * 7
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, got {len(records)}")
    for record in records:
        audit = record["final_hard_feasibility"]
        if not audit["binary"] or not audit["feasible"]:
            raise ValueError("hard-feasibility gate failed")
        if len(record["final_active_project_link_indices"]) != audit["active_links"]:
            raise ValueError("link identity gate failed")


def evaluate(output: Path, num_graphs: int, force: bool) -> None:
    if output.is_file() and not force:
        try:
            _validate(output, num_graphs)
            print(f"validated raw result; reuse {output}", flush=True)
            return
        except ValueError as error:
            print(f"cached result rejected: {error}", flush=True)
    _run([
        sys.executable, "-u", str(FIGURE_DIR / "run_snapshot_ablation.py"),
        "--num-graphs", str(num_graphs), "--output", str(output),
    ])
    _validate(output, num_graphs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    artifacts = {
        str(path.relative_to(FIGURE_DIR)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in FIGURE_DIR.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    raw = RAW_DIR / "snapshot_ablation_N20.json"
    settings = json.loads(raw.read_text(encoding="utf-8"))["settings"] if raw.is_file() else None
    (FIGURE_DIR / "manifest.json").write_text(json.dumps({
        "schema": "tnse_figure_manifest_v1", "figure": "fig03_snapshot_ablation",
        "created_at": datetime.now().astimezone().isoformat(),
        "frozen_git_revision": "ad2ee67d76e05effa4953db18ab677502ddd3ba1",
        "settings": settings, "artifacts": artifacts,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "plot", "smoke", "all"))
    parser.add_argument("--num-graphs", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "smoke":
        evaluate(SMOKE_DIR / "snapshot_ablation_N20.json", 2, True)
    elif args.stage == "evaluate":
        evaluate(RAW_DIR / "snapshot_ablation_N20.json", args.num_graphs, args.force)
    elif args.stage == "plot":
        _run([sys.executable, "-u", str(FIGURE_DIR / "plot.py")])
    else:
        evaluate(SMOKE_DIR / "snapshot_ablation_N20.json", 2, True)
        evaluate(RAW_DIR / "snapshot_ablation_N20.json", args.num_graphs, args.force)
        _run([sys.executable, "-u", str(FIGURE_DIR / "plot.py")])
    write_manifest()


if __name__ == "__main__":
    main()
