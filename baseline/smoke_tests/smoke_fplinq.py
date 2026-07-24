#!/usr/bin/env python3
"""Minimal pure numerical smoke test for the FPLinQ adapter."""

import json
from pathlib import Path
import sys

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.fplinq_numpy import binarize_fplinq, optimize_fplinq


received = np.array(
    [[2.0, 0.25, 0.05], [0.12, 1.8, 0.18], [0.04, 0.15, 1.5]],
    dtype=np.float64,
)
weights = np.array([3.0, 1.0, 2.0])
allocation, diagnostics = optimize_fplinq(
    received,
    noise_power=0.2,
    weights=weights,
    iterations=100,
    return_diagnostics=True,
)
action = binarize_fplinq(allocation)
assert allocation.shape == (3,)
assert action.shape == (3,)
assert np.isfinite(allocation).all()
assert np.all((action == 0.0) | (action == 1.0))
print(json.dumps({
    "status": "pass",
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "iterations": diagnostics.iterations,
    "allocation": allocation.tolist(),
    "action": action.astype(int).tolist(),
    "matrix_semantics": "received_power_at_rx_i_from_tx_j",
}))
