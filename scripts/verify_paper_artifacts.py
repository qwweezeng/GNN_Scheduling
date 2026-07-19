#!/usr/bin/env python3
"""Fast provenance audit for paper datasets and optional GAT checkpoints.

This check reads manifests instead of unpickling multi-gigabyte datasets.  The
generation/training scripts still perform the stricter payload validation when
they actually reuse an artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    paper_dataset_paths,
)
from utils.project_setup import (  # noqa: E402
    channel_checkpoint_path,
    load_config,
)


def _csv_ints(text):
    return [int(item) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num-uavs", default="20,30,40,50")
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    failures = []
    for num_uav in _csv_ints(args.num_uavs):
        config = dict(base, NUM_UAV=num_uav)
        seeds = {"train": 10086, "val": 10087, "test": 10088}
        sizes = {"train": 500, "val": 128, "test": 128}
        expected = expected_dataset_metadata(config, num_uav, seeds, sizes)
        train_val, test = paper_dataset_paths(
            Path("datasets/paper"), num_uav, seeds, sizes
        )
        manifest_path = train_val.parent / "manifest.json"
        if not manifest_path.exists():
            failures.append(f"N={num_uav}: missing {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("metadata") != expected:
            failures.append(f"N={num_uav}: manifest metadata mismatch")
        for name, path in (("train_val", train_val), ("test", test)):
            recorded = manifest.get("files", {}).get(name, {})
            if not path.exists():
                failures.append(f"N={num_uav}: missing {path}")
            elif recorded.get("bytes") != path.stat().st_size:
                failures.append(f"N={num_uav}: size mismatch for {path}")

        checkpoint_path = channel_checkpoint_path(config, num_uav)
        if args.require_models or checkpoint_path.exists():
            if not checkpoint_path.exists():
                failures.append(f"N={num_uav}: missing {checkpoint_path}")
            else:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                required = {
                    "model_version": "channel_gat_v1",
                    "tx_power_dbm": 10.0,
                    "csi_mode": "perfect",
                    "channel_model": "rician",
                    "rician_k_db": 5.0,
                    "num_uav": num_uav,
                    "integrity_status": "passed",
                }
                mismatch = {
                    key: (checkpoint.get(key), value)
                    for key, value in required.items()
                    if checkpoint.get(key) != value
                }
                if mismatch:
                    failures.append(
                        f"N={num_uav}: checkpoint metadata mismatch {mismatch}"
                    )
        print(
            f"N={num_uav}: dataset fingerprint={expected['fingerprint'][:12]}..."
        )
    if failures:
        raise SystemExit("Artifact audit failed:\n- " + "\n- ".join(failures))
    print("Artifact audit passed.")


if __name__ == "__main__":
    main()
