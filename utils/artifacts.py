"""Strict artifact provenance for paper datasets and checkpoints.

An artifact is reusable only when its embedded metadata agrees with the
requested physical scenario and split sizes.  A matching filename alone is
never sufficient.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping


DATASET_SCHEMA = "tnse_channel_dataset_v1"


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()


def paper_physics_metadata(config: Mapping[str, Any], num_uav: int) -> dict[str, Any]:
    """Return the physical fields that define channel-dataset semantics."""
    return {
        "num_uav": int(num_uav),
        "area_width_m": float(config["AREA_WIDTH"]),
        "height_range_m": [float(v) for v in config["HEIGHT_RANGE"]],
        "minimum_uav_distance_m": float(config["MIN_DISTANCE"]),
        "tx_power_dbm": float(config["TX_POWER"]),
        "bandwidth_hz": float(config["BANDWIDTH"]),
        "carrier_frequency_hz": float(config["CARRIER_FREQ"]),
        "noise_psd_dbm_hz": float(config["NOISE_PSD"]),
        "snr_threshold_db": float(config["SNR_THRES"]),
        "snr_gap_db": float(config["SNR_GAP_DB"]),
        "channel_model": str(config["CHANNEL_MODEL"]).lower(),
        "rician_k_db": float(config["RICIAN_K_DB"]),
        "csi_mode": str(config["CSI_MODE"]),
        "normalized_small_scale_mean_power": 1.0,
    }


def paper_dataset_paths(root: Path, num_uav: int, seeds: Mapping[str, int], sizes: Mapping[str, int]):
    directory = root / "rician_K5_tx10" / "perfect" / f"N{int(num_uav)}"
    train_val = directory / (
        f"train_seed{int(seeds['train'])}_n{int(sizes['train'])}__"
        f"val_seed{int(seeds['val'])}_n{int(sizes['val'])}.pkl"
    )
    test = directory / f"test_seed{int(seeds['test'])}_n{int(sizes['test'])}.pkl"
    return train_val, test


def paper_physics_metadata_quantized(
    config: Mapping[str, Any], num_uav: int, csi_mode: str, q_bits: int
) -> dict[str, Any]:
    """Return physics metadata for large-scale / quantized CSI variants.

    Unlike the base paper_physics_metadata() which hard-codes csi_mode="perfect",
    this function records the observed CSI mode used during training so that
    checkpoints are provenance-tracked and not accidentally reused across
    different CSI conditions.
    """
    base = paper_physics_metadata(config, num_uav)
    base["csi_mode"] = str(csi_mode)
    if csi_mode == "quantized":
        base["quantization_bits"] = int(q_bits)
    elif csi_mode == "large_scale":
        base["quantization_bits"] = None
    return base


def _csi_subdirectory(csi_mode: str, q_bits: int | None) -> str:
    if csi_mode == "large_scale":
        return "large_scale"
    if csi_mode == "quantized":
        return f"quantized_{int(q_bits)}bit"
    return "perfect"


def paper_dataset_paths_quantized(
    root: Path,
    num_uav: int,
    seeds: Mapping[str, int],
    sizes: Mapping[str, int],
    csi_mode: str,
    q_bits: int | None = None,
):
    subdir = _csi_subdirectory(csi_mode, q_bits)
    directory = root / "rician_K5_tx10" / subdir / f"N{int(num_uav)}"
    train_val = directory / (
        f"train_seed{int(seeds['train'])}_n{int(sizes['train'])}__"
        f"val_seed{int(seeds['val'])}_n{int(sizes['val'])}.pkl"
    )
    test = directory / f"test_seed{int(seeds['test'])}_n{int(sizes['test'])}.pkl"
    return train_val, test


def expected_dataset_metadata_quantized(
    config: Mapping[str, Any],
    num_uav: int,
    split_seeds: Mapping[str, int],
    split_sizes: Mapping[str, int],
    csi_mode: str,
    q_bits: int | None = None,
) -> dict[str, Any]:
    physics = paper_physics_metadata_quantized(config, num_uav, csi_mode, q_bits)
    semantic = {
        "schema": DATASET_SCHEMA,
        "physics": physics,
        "split_seeds": {key: int(value) for key, value in split_seeds.items()},
        "split_sizes": {key: int(value) for key, value in split_sizes.items()},
    }
    return {**semantic, "fingerprint": metadata_fingerprint(semantic)}


def expected_dataset_metadata(
    config: Mapping[str, Any],
    num_uav: int,
    split_seeds: Mapping[str, int],
    split_sizes: Mapping[str, int],
) -> dict[str, Any]:
    physics = paper_physics_metadata(config, num_uav)
    semantic = {
        "schema": DATASET_SCHEMA,
        "physics": physics,
        "split_seeds": {key: int(value) for key, value in split_seeds.items()},
        "split_sizes": {key: int(value) for key, value in split_sizes.items()},
    }
    return {**semantic, "fingerprint": metadata_fingerprint(semantic)}


def validate_dataset_payload(
    payload: Mapping[str, Any], expected: Mapping[str, Any], required_splits: tuple[str, ...]
) -> list[str]:
    """Return mismatch descriptions; an empty list means the cache is valid."""
    errors: list[str] = []
    metadata = payload.get("metadata", {})
    for key in ("schema", "physics", "split_seeds", "split_sizes", "fingerprint"):
        if metadata.get(key) != expected.get(key):
            errors.append(f"metadata.{key} mismatch")
    for split in required_splits:
        key = f"{split}_set"
        if key not in payload:
            errors.append(f"missing {key}")
            continue
        expected_size = int(expected["split_sizes"][split])
        if len(payload[key]) != expected_size:
            errors.append(
                f"{key} has {len(payload[key])} samples, expected {expected_size}"
            )
    return errors


def load_validated_dataset(path: Path, expected: Mapping[str, Any], required_splits: tuple[str, ...]):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    errors = validate_dataset_payload(payload, expected, required_splits)
    if errors:
        raise ValueError(f"Dataset cache {path} is incompatible: {errors}")
    return payload


def atomic_pickle_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def write_dataset_manifest(
    directory: Path,
    metadata: Mapping[str, Any],
    train_val_path: Path,
    test_path: Path,
) -> Path:
    manifest = {
        "metadata": metadata,
        "files": {
            "train_val": {
                "path": str(train_val_path),
                "bytes": train_val_path.stat().st_size,
            },
            "test": {
                "path": str(test_path),
                "bytes": test_path.stat().st_size,
            },
        },
    }
    output = directory / "manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


__all__ = [
    "DATASET_SCHEMA",
    "atomic_pickle_dump",
    "expected_dataset_metadata",
    "expected_dataset_metadata_quantized",
    "load_validated_dataset",
    "paper_dataset_paths",
    "paper_dataset_paths_quantized",
    "paper_physics_metadata",
    "paper_physics_metadata_quantized",
    "validate_dataset_payload",
    "write_dataset_manifest",
]

