#!/usr/bin/env python3
"""Staged, paired trace pipeline for TNSE Fig. 4 dynamic queues."""

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
RAW_DIR = FIGURE_DIR / "data/raw/formal_N20"
SMOKE_DIR = FIGURE_DIR / "data/smoke/N20"
TRACE_DIR = FIGURE_DIR / "data/traces"
TRACE = TRACE_DIR / "perfect_rician_N20_10k_5seeds.pt"
SMOKE_TRACE = TRACE_DIR / "smoke_perfect_rician_N20.pt"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-scheduling-mpl")

METHODS = (
    "GAT-Assisted MarginalMP,Direct-Link MarginalMP,Direct-Link LocalMax,"
    "Direct-Link Join,Hard-Conflict LGS,Random,Weighted FPLinQ,Centralized Greedy"
)
FORMAL_LAMBDAS = "0.5,1,2,2.2,2.4,2.6,2.8,3,3.2"
FORMAL_SEEDS = "42,43,44,45,46"


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _prepare_trace(path: Path, slots: int, lambdas: str, seeds: str) -> None:
    if path.is_file():
        return
    code = (
        "from pathlib import Path; from utils.fixed_csi_benchmark import build_fixed_csi_trace_cache, save_trace_cache; "
        "from utils.project_setup import load_config; "
        f"c=load_config('configs/paper.yaml'); c.update({{'NUM_UAV':20,'TX_POWER':10.0,'CHANNEL_MODEL':'rician','RICIAN_K_DB':5.0,'CSI_MODE':'perfect'}}); "
        f"p=build_fixed_csi_trace_cache(c,{slots},[{lambdas}],[{seeds}]); "
        f"save_trace_cache(p,Path(r'{path}'))"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-c", code])


def _validate(directory: Path, slots: int, seeds: str, methods: str) -> None:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if int(config["traffic"]["num_slots"]) != int(slots):
        raise ValueError("slot-count mismatch")
    if config["traffic"]["mc_seeds"] != [int(x) for x in seeds.split(",")]:
        raise ValueError("paired-seed mismatch")
    expected = methods.split(",")
    # The benchmark registry has its own deterministic display order.  Fairness
    # requires the same method *set* and paired traces, not caller-list order.
    if set(config["methods"]) != set(expected) or len(config["methods"]) != len(expected):
        raise ValueError("method set mismatch")
    if config["physical_setting"]["channel_model"] != "rician" or float(config["physical_setting"]["rician_k_db"]) != 5.0:
        raise ValueError("Rician K=5 gate failed")
    for method in expected:
        for lam in config["traffic"]["arrival_rates"]:
            item = metrics.get(f"{method}|lambda={float(lam):g}")
            if item is None or int(item.get("num_mc_runs", 0)) != len(config["traffic"]["mc_seeds"]):
                raise ValueError(f"missing paired aggregate for {method}, Lambda={lam}")


def run(directory: Path, trace: Path, slots: int, lambdas: str, seeds: str, methods: str, force: bool) -> None:
    if (directory / "config.json").is_file() and not force:
        try:
            _validate(directory, slots, seeds, methods)
            print(f"validated raw result; reuse {directory}", flush=True)
            return
        except ValueError as error:
            print(f"cached result rejected: {error}", flush=True)
    _prepare_trace(trace, slots, lambdas, seeds)
    _run([
        sys.executable, "-u", "experiments/exp_fixed_csi_qos.py",
        "--num_uav", "20", "--num_slots", str(slots),
        "--arrival_rates", lambdas, "--mc_seeds", seeds,
        "--rounds", "4", "--channel_model", "rician", "--rician_k_db", "5",
        "--trace_cache", str(trace), "--methods", methods,
        "--output_dir", str(directory),
    ])
    _validate(directory, slots, seeds, methods)


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
    formal = RAW_DIR / "config.json"
    smoke = SMOKE_DIR / "config.json"
    (FIGURE_DIR / "manifest.json").write_text(json.dumps({
        "schema": "tnse_figure_manifest_v1", "figure": "fig04_dynamic_queue",
        "created_at": datetime.now().astimezone().isoformat(),
        "frozen_git_revision": "ad2ee67d76e05effa4953db18ab677502ddd3ba1",
        "formal_config": json.loads(formal.read_text(encoding="utf-8")) if formal.is_file() else None,
        "smoke_config": json.loads(smoke.read_text(encoding="utf-8")) if smoke.is_file() else None,
        "artifacts": artifacts,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "plot", "smoke", "all"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        _prepare_trace(TRACE, 10000, FORMAL_LAMBDAS, FORMAL_SEEDS)
    elif args.stage == "smoke":
        # Minimal paired interface check; it is deliberately not formal evidence.
        run(SMOKE_DIR, SMOKE_TRACE, 50, "0.5,2", "42,43",
            "GAT-Assisted MarginalMP,Direct-Link MarginalMP,Hard-Conflict LGS,Centralized Greedy", True)
    elif args.stage == "run":
        run(RAW_DIR, TRACE, 10000, FORMAL_LAMBDAS, FORMAL_SEEDS, METHODS, args.force)
    elif args.stage == "plot":
        _run([sys.executable, "-u", str(FIGURE_DIR / "plot.py")])
    else:
        run(SMOKE_DIR, SMOKE_TRACE, 50, "0.5,2", "42,43",
            "GAT-Assisted MarginalMP,Direct-Link MarginalMP,Hard-Conflict LGS,Centralized Greedy", True)
        run(RAW_DIR, TRACE, 10000, FORMAL_LAMBDAS, FORMAL_SEEDS, METHODS, args.force)
        _run([sys.executable, "-u", str(FIGURE_DIR / "plot.py")])
    write_manifest()


if __name__ == "__main__":
    main()
