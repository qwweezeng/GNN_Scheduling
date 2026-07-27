#!/usr/bin/env python3
"""Reproducible staged pipeline for TNSE Fig. 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
RAW_DIR = FIGURE_DIR / "data/raw"
FIGURES_DIR = FIGURE_DIR / "figures"
CHECKPOINT_DIR = FIGURE_DIR / "checkpoints"
FROZEN_REVISION = "ad2ee67d76e05effa4953db18ab677502ddd3ba1"
DEFAULT_NODES = (20, 30, 40, 50)
DEFAULT_TRAIN_SIZES = (50, 100, 200, 300, 500)
SPLIT_SEEDS = {"train": 10086, "val": 10087, "test": 10088}
POOL_SIZES = {"train": 300, "val": 128, "test": 128}

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "gnn-scheduling-mpl")
)
sys.path.insert(0, str(REPO_ROOT))

from models.model import calculate_achievable_rate  # noqa: E402
from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.feasibility import (  # noqa: E402
    decode_endpoint_mutual_matching,
    shared_uav_action_audit,
)
from utils.project_setup import (  # noqa: E402
    channel_checkpoint_path,
    load_channel_gat,
    load_channel_gat_checkpoint,
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


def _pool_metadata(num_uav: int) -> dict:
    return expected_dataset_metadata(
        _paper_config(num_uav), num_uav, SPLIT_SEEDS, POOL_SIZES
    )


def _pool_paths(num_uav: int) -> tuple[Path, Path]:
    return paper_dataset_paths(
        REPO_ROOT / "datasets/paper", num_uav, SPLIT_SEEDS, POOL_SIZES
    )


def _subset_report(num_uav: int, train_size: int) -> Path:
    return RAW_DIR / f"training_N{num_uav}_n{train_size}.json"


def _subset_checkpoint(num_uav: int, train_size: int) -> Path:
    return CHECKPOINT_DIR / f"gat_N{num_uav}_train{train_size}.pth"


def _production_report(num_uav: int) -> Path:
    return REPO_ROOT / "results/paper/channel_gat_training" / (
        f"rician_channel_gat_N{num_uav}_K5.json"
    )


def _report_path(num_uav: int, train_size: int) -> Path:
    return (
        _production_report(num_uav)
        if int(train_size) == 500
        else _subset_report(num_uav, train_size)
    )


def _checkpoint_path(num_uav: int, train_size: int) -> Path:
    return (
        REPO_ROOT / channel_checkpoint_path(_paper_config(num_uav), num_uav)
        if int(train_size) == 500
        else _subset_checkpoint(num_uav, train_size)
    )


def _repo_relative(path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        return str(value)
    return str(value.relative_to(REPO_ROOT))


def prepare(nodes: list[int]) -> None:
    missing = []
    for num_uav in nodes:
        train_val, test = _pool_paths(num_uav)
        if not train_val.is_file() or not test.is_file():
            missing.append(num_uav)
            continue
        expected = _pool_metadata(num_uav)
        load_validated_dataset(train_val, expected, ("train", "val"))
        load_validated_dataset(test, expected, ("test",))
        print(f"[N={num_uav}] canonical 300/128/128 pool verified", flush=True)
    if missing:
        _run([
            sys.executable,
            "-u",
            "scripts/prepare_paper_datasets.py",
            "--num-uavs",
            ",".join(str(value) for value in missing),
        ])
        prepare(missing)


def _validate_report(num_uav: int, train_size: int) -> None:
    path = _report_path(num_uav, train_size)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fp = _pool_metadata(num_uav)["fingerprint"]
    expected = {
        "num_uav": int(num_uav),
        "tx_power_dbm": 10.0,
        "channel_model": "rician",
        "rician_k_db": 5.0,
        "csi_mode": "perfect",
        "dataset_fingerprint": expected_fp,
        "integrity_status": "passed",
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if int(payload.get("config", {}).get("TRAIN_SIZE", -1)) != int(train_size):
        mismatches["config.TRAIN_SIZE"] = (
            payload.get("config", {}).get("TRAIN_SIZE"), int(train_size)
        )
    if len(payload.get("per_epoch_logs", [])) < 1:
        mismatches["per_epoch_logs"] = (0, ">=1")
    if not bool(payload.get("output_audit", {}).get("integrity_passed", False)):
        mismatches["output_audit.integrity_passed"] = (False, True)
    if train_size != 500:
        subset = payload.get("training_subset", {})
        subset_expected = {
            "rule": "first_n_prefix_of_canonical_training_pool",
            "selected_size": int(train_size),
            "pool_size": 500,
            "pool_dataset_fingerprint": expected_fp,
        }
        if subset != subset_expected:
            mismatches["training_subset"] = (subset, subset_expected)
    if mismatches:
        raise ValueError(f"training report mismatch in {path}: {mismatches}")


def _validate_checkpoint(num_uav: int, train_size: int) -> None:
    fingerprint = _pool_metadata(num_uav)["fingerprint"]
    if train_size == 500:
        load_channel_gat(
            _paper_config(num_uav), num_uav, device="cpu",
            expected_dataset_fingerprint=fingerprint,
        )
    else:
        load_channel_gat_checkpoint(
            _paper_config(num_uav), _checkpoint_path(num_uav, train_size),
            num_uav, device="cpu", expected_dataset_fingerprint=fingerprint,
        )


def train(
    nodes: list[int], train_sizes: list[int], *, epochs: int, force: bool
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for num_uav in nodes:
        for train_size in train_sizes:
            if train_size == 500:
                _validate_report(num_uav, train_size)
                _validate_checkpoint(num_uav, train_size)
                print(
                    f"[N={num_uav}, n=500] production training artifacts verified",
                    flush=True,
                )
                continue
            if not force:
                try:
                    _validate_report(num_uav, train_size)
                    _validate_checkpoint(num_uav, train_size)
                    print(
                        f"[N={num_uav}, n={train_size}] subset checkpoint reused",
                        flush=True,
                    )
                    continue
                except (FileNotFoundError, ValueError, RuntimeError) as error:
                    print(
                        f"[N={num_uav}, n={train_size}] training required: {error}",
                        flush=True,
                    )
            _run([
                sys.executable,
                "-u",
                "train/train_channel_gat_rician.py",
                "--num_uav", str(num_uav),
                "--train_size", str(train_size),
                "--prefix_pool_size", "500",
                "--val_size", "128",
                "--test_size", "128",
                "--epochs", str(epochs),
                "--seed", "10086",
                "--val_seed", "10087",
                "--test_seed", "10088",
                "--checkpoint_out", str(_subset_checkpoint(num_uav, train_size)),
                "--report_out", str(_subset_report(num_uav, train_size)),
            ])
            _validate_report(num_uav, train_size)
            _validate_checkpoint(num_uav, train_size)


def _load_model(num_uav: int, train_size: int):
    fingerprint = _pool_metadata(num_uav)["fingerprint"]
    if train_size == 500:
        return load_channel_gat(
            _paper_config(num_uav), num_uav, device="cpu",
            expected_dataset_fingerprint=fingerprint,
        )
    return load_channel_gat_checkpoint(
        _paper_config(num_uav), _checkpoint_path(num_uav, train_size),
        num_uav, device="cpu", expected_dataset_fingerprint=fingerprint,
    )


def _evaluate_one(num_uav: int, train_size: int) -> dict:
    model, checkpoint = _load_model(num_uav, train_size)
    _, test_path = _pool_paths(num_uav)
    expected = _pool_metadata(num_uav)
    states = load_validated_dataset(test_path, expected, ("test",))["test_set"]
    records = []
    with torch.no_grad():
        for sample_index, state in enumerate(states):
            probability, _ = model(
                state.vertex_features[:, :1], state.edge_index, state.edge_weights
            )
            score = probability.reshape(-1).float().cpu()
            candidate = (score >= 0.5).float()
            raw_audit = shared_uav_action_audit(candidate.reshape(-1, 1), state.links)
            action, decoder_audit = decode_endpoint_mutual_matching(
                candidate, state.links, priorities=score, max_rounds=1
            )
            final_audit = shared_uav_action_audit(action.reshape(-1, 1), state.links)
            if not final_audit["binary"] or not final_audit["feasible"]:
                raise RuntimeError("endpoint-mutual decoder emitted invalid action")
            rate = float(calculate_achievable_rate(
                action.reshape(-1, 1), state, binary=True, detach_hard=True
            ).sum())
            records.append({
                "sample_index": sample_index,
                "num_links": int(state.links.size(0)),
                "sum_spectral_efficiency_bit_s_hz": rate,
                "aggregate_phy_rate_mbps": rate * float(state.bandwidth) / 1e6,
                "candidate_active_fraction": float(candidate.mean()),
                "final_active_fraction": float(action.float().mean()),
                "candidate_endpoint_excess": int(raw_audit["endpoint_excess_count"]),
                "candidate_feasible": bool(raw_audit["feasible"]),
                "final_active_link_indices": torch.nonzero(
                    action.reshape(-1), as_tuple=True
                )[0].tolist(),
                "hard_feasibility": decoder_audit,
            })
    rates = np.asarray([row["aggregate_phy_rate_mbps"] for row in records])
    active = np.asarray([row["final_active_fraction"] for row in records])
    return {
        "schema": "tnse_fig02_subset_evaluation_v1",
        "num_uav": num_uav,
        "train_size": train_size,
        "train_pool_size": 500,
        "subset_rule": "first_n_prefix_of_canonical_training_pool",
        "split_sizes": POOL_SIZES,
        "split_seeds": SPLIT_SEEDS,
        "dataset_fingerprint": expected["fingerprint"],
        "test_dataset_path": str(test_path.relative_to(REPO_ROOT)),
        "checkpoint": _repo_relative(checkpoint),
        "decoder": "one_round_endpoint_mutual_handshake",
        "service_channel": "current true instantaneous Rician channel",
        "num_test_samples": len(records),
        "mean_aggregate_phy_rate_mbps": float(rates.mean()),
        "std_aggregate_phy_rate_mbps": float(rates.std(ddof=0)),
        "sem_aggregate_phy_rate_mbps": float(rates.std(ddof=1) / np.sqrt(len(rates))),
        "mean_final_active_fraction": float(active.mean()),
        "all_final_actions_feasible": True,
        "per_sample_records": records,
    }


def evaluate(nodes: list[int], train_sizes: list[int], *, force: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for num_uav in nodes:
        for train_size in train_sizes:
            _validate_report(num_uav, train_size)
            _validate_checkpoint(num_uav, train_size)
            output = RAW_DIR / f"evaluation_N{num_uav}_n{train_size}.json"
            if output.is_file() and not force:
                cached = json.loads(output.read_text(encoding="utf-8"))
                if (
                    cached.get("dataset_fingerprint")
                    == _pool_metadata(num_uav)["fingerprint"]
                    and int(cached.get("num_test_samples", -1)) == 128
                    and bool(cached.get("all_final_actions_feasible", False))
                ):
                    print(
                        f"[N={num_uav}, n={train_size}] evaluation reused",
                        flush=True,
                    )
                    continue
            payload = _evaluate_one(num_uav, train_size)
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"[N={num_uav}, n={train_size}] "
                f"test={payload['mean_aggregate_phy_rate_mbps']:.3f} Mbit/s",
                flush=True,
            )


def plot(nodes: list[int], train_sizes: list[int]) -> None:
    _run([
        sys.executable,
        "-u",
        str(FIGURE_DIR / "plot.py"),
        "--nodes", ",".join(str(value) for value in nodes),
        "--train-sizes", ",".join(str(value) for value in train_sizes),
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


def write_manifest(nodes: list[int], train_sizes: list[int], command: list[str]) -> None:
    artifacts = {}
    for directory in (FIGURE_DIR / "data", CHECKPOINT_DIR, FIGURES_DIR):
        if directory.exists():
            for path in sorted(value for value in directory.rglob("*") if value.is_file()):
                artifacts[str(path.relative_to(REPO_ROOT))] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
    sources = {}
    for num_uav in nodes:
        sources[str(num_uav)] = {
            "pool_fingerprint": _pool_metadata(num_uav)["fingerprint"],
            "trainings": {
                str(size): {
                    "report": str(_report_path(num_uav, size).relative_to(REPO_ROOT)),
                    "checkpoint": str(_checkpoint_path(num_uav, size).relative_to(REPO_ROOT)),
                }
                for size in train_sizes
                if _report_path(num_uav, size).exists()
                and _checkpoint_path(num_uav, size).exists()
            },
        }
    manifest = {
        "schema": "tnse_figure_manifest_v1",
        "figure": "fig02_training_audit",
        "created_at": datetime.now().astimezone().isoformat(),
        "frozen_git_revision": FROZEN_REVISION,
        "current_git_revision": _git_output("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git_output("status", "--short")),
        "command": command,
        "python_executable": sys.executable,
        "nodes": nodes,
        "train_sizes": train_sizes,
        "training_subset_rule": "first_n_prefix_of_canonical_500_sample_pool",
        "split_sizes": POOL_SIZES,
        "split_seeds": SPLIT_SEEDS,
        "physics": {
            "tx_power_dbm": 10.0,
            "channel_model": "rician",
            "rician_k_db": 5.0,
            "carrier_frequency_hz": 2.4e9,
            "bandwidth_hz": 5.0e6,
            "csi_observed": "perfect instantaneous",
            "service_channel": "current true instantaneous Rician channel",
            "shared_uav_constraint": "single_radio_half_duplex_matching",
            "final_decoder": "one_round_endpoint_mutual_handshake",
        },
        "sources": sources,
        "local_artifacts": artifacts,
    }
    (FIGURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "plot", "all"))
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--train-sizes", default="50,100,200,300,500")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = _csv_ints(args.nodes)
    train_sizes = _csv_ints(args.train_sizes)
    unknown_sizes = sorted(set(train_sizes) - set(DEFAULT_TRAIN_SIZES))
    if unknown_sizes:
        raise ValueError(f"unsupported formal training sizes: {unknown_sizes}")
    if args.stage in {"prepare", "all"}:
        prepare(nodes)
    if args.stage in {"train", "all"}:
        train(nodes, train_sizes, epochs=args.epochs, force=args.force_train)
    if args.stage in {"evaluate", "all"}:
        evaluate(nodes, train_sizes, force=args.force_evaluate)
    if args.stage in {"plot", "all"}:
        plot(nodes, train_sizes)
    write_manifest(nodes, train_sizes, [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
