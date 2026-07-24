#!/usr/bin/env python3
"""Load a retained DistGCN TF1 checkpoint and verify link-index roundtrip."""

import argparse
import json
from pathlib import Path
import sys

import cnfgen
import dwave_networkx
import igraph
import natsort
import networkx
import numpy as np
import pandas
import pulp
import scipy
import spektral
import tensorflow as tf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from baseline.distgcn_adapter import LinkIndexRoundTrip
from baseline.distgcn_checkpoint import restore_and_inspect_checkpoint


DEFAULT_CHECKPOINT = REPO / (
    "baseline/distgcn-main/model/"
    "result_DQNBA_deep_ld1_c32_l3_cheb1_diver1_mwis_dqn"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
args = parser.parse_args()

inspection = restore_and_inspect_checkpoint(args.checkpoint)
mapping = LinkIndexRoundTrip.from_project_indices(7, [5, 1, 6, 0])
project_values = np.arange(7, dtype=np.float32)
third_party_values = mapping.project_to_third_party(project_values)
project_action = mapping.third_party_to_project_action([1, 0, 1, 0])
assert third_party_values.tolist() == [5, 1, 6, 0]
assert project_action.tolist() == [0, 0, 0, 0, 0, 1, 1]

print(json.dumps({
    "status": "pass",
    "checkpoint": inspection["checkpoint_prefix"],
    "checkpoint_variable_count": inspection["variable_count"],
    "link_index_roundtrip": mapping.metadata(),
    "versions": {
        "tensorflow": tf.__version__,
        "spektral": spektral.__version__,
        "networkx": networkx.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "dwave_networkx": dwave_networkx.__version__,
        "cnfgen": getattr(cnfgen, "__version__", "unknown"),
        "pulp": pulp.__version__,
        "igraph": igraph.__version__,
        "natsort": natsort.__version__,
    },
}))
