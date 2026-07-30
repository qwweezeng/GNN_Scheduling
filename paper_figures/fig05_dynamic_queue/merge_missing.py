#!/usr/bin/env python3
"""Merge missing N=50 Fig. 5 aggregate points without replacing existing data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data/raw/deadline_2000ms/N50"
SOURCE = ROOT / "data/work/deadline_2000ms_N50_missing10"
ALL_RATES = (0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0)
MISSING_RATES = frozenset(ALL_RATES[:10])
METHODS = frozenset((
    "Random", "Centralized Greedy", "Weighted FPLinQ",
    "Direct-Link LocalMax", "GAT-Assisted LocalMax",
    "Direct-Link MarginalMP", "GAT-Assisted MarginalMP",
))


def key_rate(key: str) -> float:
    return float(key.rsplit("|lambda=", 1)[1].split("|", 1)[0])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".merge_tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def merge_metric_file(name: str) -> None:
    target = load_json(TARGET / name)
    source = load_json(SOURCE / name)
    old = dict(target)
    additions = {
        key: value for key, value in source.items()
        if key_rate(key) in MISSING_RATES
    }
    expected = len(METHODS) * len(MISSING_RATES)
    if len(additions) != expected:
        raise RuntimeError(f"{name}: expected {expected} additions, got {len(additions)}")
    existing_additions = {key: value for key, value in additions.items() if key in target}
    if any(target[key] != value for key, value in existing_additions.items()):
        raise RuntimeError(f"{name}: existing result conflicts with source")
    additions = {key: value for key, value in additions.items() if key not in target}
    if any(key.split("|lambda=", 1)[0] not in METHODS for key in additions):
        raise RuntimeError(f"{name}: unexpected method in additions")
    merged = dict(target)
    merged.update(additions)
    atomic_json(TARGET / name, merged)
    check = load_json(TARGET / name)
    for key, value in old.items():
        if check.get(key) != value:
            raise RuntimeError(f"{name}: existing key changed: {key}")
    print(f"{name}: preserved {len(old)}, added {len(additions)}, total {len(check)}")


def merge_trajectories() -> None:
    target_path = TARGET / "trajectories.npz"
    source_path = SOURCE / "trajectories.npz"
    with np.load(target_path, allow_pickle=False) as target_file, np.load(source_path, allow_pickle=False) as source_file:
        old = {key: target_file[key] for key in target_file.files}
        additions = {
            key: source_file[key] for key in source_file.files
            if any(f"__lambda_{str(rate).replace('.', 'p')}__" in key for rate in MISSING_RATES)
        }
    metrics_per_task = 10
    expected = len(METHODS) * len(MISSING_RATES) * metrics_per_task
    if len(additions) != expected:
        raise RuntimeError(f"trajectories.npz: expected {expected} additions, got {len(additions)}")
    if set(old) & set(additions):
        raise RuntimeError("trajectories.npz: refusing to overwrite an existing key")
    merged = dict(old)
    merged.update(additions)
    temp = target_path.with_name(target_path.stem + ".merge_tmp.npz")
    np.savez_compressed(temp, **merged)
    os.replace(temp, target_path)
    with np.load(target_path, allow_pickle=False) as check_file:
        if any(not np.array_equal(check_file[key], value) for key, value in old.items()):
            raise RuntimeError("trajectories.npz: existing array changed")
    print(f"trajectories.npz: preserved {len(old)}, added {len(additions)}, total {len(merged)}")


def main() -> None:
    target_config = load_json(TARGET / "config.json")
    source_config = load_json(SOURCE / "config.json")
    if target_config["traffic"]["mc_seeds"] != [42] or source_config["traffic"]["mc_seeds"] != [42]:
        raise RuntimeError("refusing to merge a non-seed-42 result")
    if target_config["traffic"]["arrival_rates"] != [3.5, 4.0, 4.5, 5.0]:
        raise RuntimeError("target rates changed unexpectedly")
    if source_config["traffic"]["arrival_rates"] != list(ALL_RATES):
        raise RuntimeError("source does not contain the complete Fig. 5 grid")
    if target_config["methods"] != source_config["methods"]:
        raise RuntimeError("method configuration mismatch")
    for name in ("metrics.json", "per_seed_metrics.json"):
        merge_metric_file(name)
    merge_trajectories()
    merged_config = dict(target_config)
    merged_config["traffic"] = dict(target_config["traffic"])
    merged_config["traffic"]["arrival_rates"] = list(ALL_RATES)
    atomic_json(TARGET / "config.json", merged_config)
    print("config.json: updated arrival-rate metadata only")


if __name__ == "__main__":
    main()
