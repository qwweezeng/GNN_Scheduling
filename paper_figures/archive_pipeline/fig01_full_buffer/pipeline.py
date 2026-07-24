#!/usr/bin/env python3
"""Reproducible staged pipeline for TNSE Fig. 1."""

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
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "gnn-scheduling-mpl")
)
RAW_DIR = FIGURE_DIR / "data/raw"
SMOKE_DIR = FIGURE_DIR / "data/smoke"
FIGURES_DIR = FIGURE_DIR / "figures"
FROZEN_REVISION = "ad2ee67d76e05effa4953db18ab677502ddd3ba1"
METHODS = (
    "Random,Top-SNR,LocalMax,Local Greedy,Spatial DL,FPLinQ,Proposed GAT"
)

sys.path.insert(0, str(REPO_ROOT))

from utils.artifacts import expected_dataset_metadata, paper_dataset_paths  # noqa: E402
from utils.project_setup import (  # noqa: E402
    channel_checkpoint_path,
    load_channel_gat,
    load_config,
)


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _paper_config(num_uav: int) -> dict:
    config = load_config(REPO_ROOT / "configs/paper.yaml")
    config.update({
        "NUM_UAV": int(num_uav),
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    return config


def _fingerprint(num_uav: int) -> str:
    metadata = expected_dataset_metadata(
        _paper_config(num_uav),
        num_uav,
        {"train": 10086, "val": 10087, "test": 10088},
        {"train": 500, "val": 128, "test": 128},
    )
    return metadata["fingerprint"]


def prepare(nodes: list[int]) -> None:
    _run([
        sys.executable,
        "-u",
        "scripts/prepare_paper_datasets.py",
        "--num-uavs",
        ",".join(str(value) for value in nodes),
    ])


def _checkpoint_valid(num_uav: int) -> tuple[bool, str]:
    try:
        _, path = load_channel_gat(
            _paper_config(num_uav),
            num_uav,
            device="cpu",
            expected_dataset_fingerprint=_fingerprint(num_uav),
        )
        return True, path
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        return False, str(error)


def train(nodes: list[int], *, epochs: int, force: bool) -> None:
    for num_uav in nodes:
        valid, detail = _checkpoint_valid(num_uav)
        if valid and not force:
            print(f"[N={num_uav}] validated checkpoint; reuse {detail}", flush=True)
            continue
        print(f"[N={num_uav}] training required: {detail}", flush=True)
        _run([
            sys.executable,
            "-u",
            "train/train_channel_gat_rician.py",
            "--num_uav",
            str(num_uav),
            "--epochs",
            str(epochs),
        ])
        valid, detail = _checkpoint_valid(num_uav)
        if not valid:
            raise RuntimeError(f"N={num_uav} checkpoint failed post-train audit: {detail}")


def _validate_result(path: Path, num_uav: int, num_samples: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("num_uav", -1)) != num_uav:
        raise ValueError(f"N mismatch in {path}")
    if int(payload.get("num_samples", -1)) != num_samples:
        raise ValueError(f"sample-count mismatch in {path}")
    if payload.get("dataset_fingerprint") != _fingerprint(num_uav):
        raise ValueError(f"dataset fingerprint mismatch in {path}")
    if set(payload.get("results", {})) != set(METHODS.split(",")):
        raise ValueError(f"method set mismatch in {path}")
    for method, result in payload["results"].items():
        records = result.get("per_sample_records", [])
        if len(records) != num_samples:
            raise ValueError(f"{method} record count mismatch in {path}")
        for record in records:
            audit = record["hard_feasibility"]["projected"]
            if not audit["binary"] or not audit["feasible"]:
                raise ValueError(f"{method} returned an infeasible action in {path}")
            if len(record["active_project_link_indices"]) != audit["active_links"]:
                raise ValueError(f"{method} link-identity mismatch in {path}")
    checkpoint = payload.get("method_metadata", {}).get("Proposed GAT", {}).get(
        "checkpoint"
    )
    if not checkpoint:
        raise ValueError(f"checkpoint metadata missing in {path}")
    metadata = payload.get("method_metadata", {})
    if metadata.get("Top-SNR", {}).get("top_snr_ratio") != 0.03:
        raise ValueError(f"Top-SNR must use the fixed 3% fraction in {path}")
    if metadata.get("LocalMax", {}).get("global_top_k") is not False:
        raise ValueError(f"distributed LocalMax metadata missing in {path}")
    if metadata.get("Proposed GAT", {}).get("feasibility_decoder") != (
        "endpoint_mutual_1"
    ):
        raise ValueError(f"one-round endpoint-mutual GAT decoder missing in {path}")


def evaluate(
    nodes: list[int], *, num_samples: int, force: bool, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for num_uav in nodes:
        output = output_dir / f"benchmark_N{num_uav}.json"
        if output.is_file() and not force:
            try:
                _validate_result(output, num_uav, num_samples)
                print(f"[N={num_uav}] validated raw result; reuse {output}", flush=True)
                continue
            except ValueError as error:
                print(f"[N={num_uav}] cached result rejected: {error}", flush=True)
        _run([
            sys.executable,
            "-u",
            "experiments/compare_gat_vs_spatial_dl.py",
            "--num_uav",
            str(num_uav),
            "--num_samples",
            str(num_samples),
            "--methods",
            METHODS,
            "--output",
            str(output),
        ])
        _validate_result(output, num_uav, num_samples)


def plot(nodes: list[int]) -> None:
    _run([
        sys.executable,
        "-u",
        str(FIGURE_DIR / "plot.py"),
        "--nodes",
        ",".join(str(value) for value in nodes),
    ])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def write_manifest(nodes: list[int], command: list[str]) -> None:
    artifacts = {}
    for directory in (FIGURE_DIR / "data", FIGURES_DIR):
        if directory.exists():
            for path in sorted(value for value in directory.rglob("*") if value.is_file()):
                artifacts[str(path.relative_to(REPO_ROOT))] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
    datasets, checkpoints = {}, {}
    for num_uav in nodes:
        config = _paper_config(num_uav)
        seeds = {"train": 10086, "val": 10087, "test": 10088}
        sizes = {"train": 500, "val": 128, "test": 128}
        train_val, test = paper_dataset_paths(
            REPO_ROOT / "datasets/paper", num_uav, seeds, sizes
        )
        datasets[str(num_uav)] = {
            "fingerprint": _fingerprint(num_uav),
            "train_val_path": str(train_val.relative_to(REPO_ROOT)),
            "train_val_bytes": train_val.stat().st_size,
            "test_path": str(test.relative_to(REPO_ROOT)),
            "test_bytes": test.stat().st_size,
        }
        checkpoint = REPO_ROOT / channel_checkpoint_path(config, num_uav)
        checkpoints[str(num_uav)] = {
            "path": str(checkpoint.relative_to(REPO_ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        }
    manifest = {
        "schema": "tnse_figure_manifest_v1",
        "figure": "fig01_full_buffer",
        "created_at": datetime.now().astimezone().isoformat(),
        "frozen_git_revision": FROZEN_REVISION,
        "current_git_revision": _git_output("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git_output("status", "--short")),
        "command": command,
        "python_executable": sys.executable,
        "nodes": nodes,
        "physics": {
            "tx_power_dbm": 10.0,
            "channel_model": "rician",
            "rician_k_db": 5.0,
            "carrier_frequency_hz": 2.4e9,
            "bandwidth_hz": 5.0e6,
            "csi_observed": "perfect instantaneous",
            "service_channel": "current true instantaneous Rician channel",
            "shared_uav_constraint": "single_radio_half_duplex_matching",
            "proposed_gat_feasible_decoder": "one_round_endpoint_mutual_handshake",
            "snr_localmax": "one-hop conflict-neighbor maximum",
        },
        "split_sizes": {"train": 500, "val": 128, "test": 128},
        "split_seeds": {"train": 10086, "val": 10087, "test": 10088},
        "methods": METHODS.split(","),
        "datasets": datasets,
        "checkpoints": checkpoints,
        "local_artifacts": artifacts,
    }
    (FIGURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "plot", "smoke", "all"))
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    nodes = [int(item) for item in args.nodes.split(",") if item.strip()]
    if args.stage == "prepare":
        prepare(nodes)
    elif args.stage == "train":
        train(nodes, epochs=args.epochs, force=args.force_train)
    elif args.stage == "evaluate":
        evaluate(nodes, num_samples=args.samples, force=args.force_evaluate, output_dir=RAW_DIR)
    elif args.stage == "plot":
        plot(nodes)
        write_manifest(nodes, [sys.executable, *sys.argv])
    elif args.stage == "smoke":
        prepare([20])
        valid, detail = _checkpoint_valid(20)
        if not valid:
            raise RuntimeError(
                "N=20 smoke requires a metadata-valid Channel-GAT checkpoint; "
                f"run pipeline.py train --nodes 20 first. Detail: {detail}"
            )
        evaluate([20], num_samples=2, force=True, output_dir=SMOKE_DIR)
    else:
        prepare(nodes)
        train(nodes, epochs=args.epochs, force=args.force_train)
        evaluate([20], num_samples=2, force=True, output_dir=SMOKE_DIR)
        evaluate(nodes, num_samples=args.samples, force=args.force_evaluate, output_dir=RAW_DIR)
        plot(nodes)
        write_manifest(nodes, [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
