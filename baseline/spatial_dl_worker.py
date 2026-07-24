#!/usr/bin/env python3
"""Private subprocess worker for the isolated Spatial-DL environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from baseline.spatial_dl_adapter import run_spatial_dl_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = np.load(args.input, allow_pickle=False)
    schedules = run_spatial_dl_checkpoint(
        payload["layouts"],
        payload["valid_masks"],
        source_root=args.source_root,
        seed=args.seed,
    )
    np.savez_compressed(args.output, schedules=schedules)


if __name__ == "__main__":
    main()
