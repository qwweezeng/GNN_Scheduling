#!/usr/bin/env python3
"""Run one canonical full-link paper state per UAV scale through Spatial DL."""

import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.baseline import eval_spatial_dl  # noqa: E402
from utils.project_setup import load_config  # noqa: E402


config = load_config("configs/paper.yaml")
seeds = {"train": 10086, "val": 10087, "test": 10088}
sizes = {"train": 500, "val": 128, "test": 128}
records = []
for num_uav in (20, 30, 40, 50):
    expected = expected_dataset_metadata(config, num_uav, seeds, sizes)
    _, test_path = paper_dataset_paths(
        Path("datasets/paper"), num_uav, seeds, sizes
    )
    payload = load_validated_dataset(test_path, expected, ("test",))
    state = payload["test_set"][0]
    values, metadata = eval_spatial_dl(
        [state], seed=20260720 + num_uav, batch_size=1, return_metadata=True
    )
    records.append({
        "num_uav": num_uav,
        "num_links": int(state.links.size(0)),
        "dataset_fingerprint": expected["fingerprint"],
        "sum_spectral_efficiency": float(values[0]),
        "adapter": metadata[0],
    })

print(json.dumps({
    "status": "pass",
    "scope": "one canonical full-link test state per scale; not a paper result",
    "records": records,
}))
