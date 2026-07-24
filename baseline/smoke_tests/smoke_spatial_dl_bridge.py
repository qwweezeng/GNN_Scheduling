#!/usr/bin/env python3
"""Verify main-process to isolated Spatial-DL checkpoint execution."""

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from baseline.spatial_dl_adapter import (  # noqa: E402
    project_3d_links_to_spatial_dl,
    run_spatial_dl_checkpoint_isolated,
)


count = 120
rng = np.random.default_rng(20260720)
positions = np.column_stack((
    rng.uniform(20, 980, size=(count + 1, 2)),
    rng.uniform(50, 150, size=count + 1),
))
links = np.column_stack((np.arange(count), np.arange(1, count + 1)))
layout, mask, mapping = project_3d_links_to_spatial_dl(
    positions, links, target_link_count=128
)
source = REPO / "baseline/Spatial_Deep_Learning_for_Wireless_Scheduling-master"
schedule = run_spatial_dl_checkpoint_isolated(
    layout[None, :, :],
    mask[None, :],
    source_root=source,
    seed=20260720,
)[0]
project_action = mapping.legacy_to_project(schedule)
assert project_action.shape == (len(links),)
assert np.all(schedule[mask == 0] == 0)
print(json.dumps({
    "status": "pass",
    "bridge": "main_process_to_gnn-spatial-dl-legacy",
    "checkpoint_training_link_count": 50,
    "inference_link_count": len(mask),
    "real_link_count": int(mask.sum()),
    "active_links": int(project_action.sum()),
    "mapping": mapping.metadata(),
}))
