#!/usr/bin/env python3
"""Restore one Spatial-DL checkpoint at several inference cardinalities."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from baseline.spatial_dl_adapter import (
    project_3d_links_to_spatial_dl,
    run_spatial_dl_checkpoint,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--link-counts", default="4,100,200,500,1300")
args = parser.parse_args()

source = REPO / "baseline/Spatial_Deep_Learning_for_Wireless_Scheduling-master"
records = []
for count in [int(item) for item in args.link_counts.split(",")]:
    rng = np.random.default_rng(20260720 + count)
    positions = np.column_stack((
        rng.uniform(20, 980, size=(count + 1, 2)),
        rng.uniform(50, 150, size=count + 1),
    ))
    links = np.column_stack((
        np.arange(count, dtype=np.int64),
        np.arange(1, count + 1, dtype=np.int64),
    ))
    layout, mask, mapping = project_3d_links_to_spatial_dl(positions, links)
    schedule = run_spatial_dl_checkpoint(
        layout, mask, source_root=source, seed=20260720
    )[0]
    assert schedule.shape == (count,)
    assert np.all((schedule == 0.0) | (schedule == 1.0))
    project_action = mapping.legacy_to_project(schedule)
    assert project_action.shape == (len(links),)
    records.append({
        "link_count": count,
        "active_links": int(project_action.sum()),
        "mapping": mapping.metadata(),
    })
print(json.dumps({
    "status": "pass",
    "tensorflow": tf.__version__,
    "checkpoint": str(source / "Neural_Network_Model/Trained_Model/conv_net_model.ckpt"),
    "checkpoint_training_link_count": 50,
    "verified_inference_link_counts": [item["link_count"] for item in records],
    "projection": "drop_z_then_uniformly_scale_xy_if_needed",
    "records": records,
}))
