#!/usr/bin/env python3
"""Pre-generate paired fixed-CSI channel and arrival traces.

This is data preparation, not a paper experiment.  A cache stores one channel
graph per Monte Carlo seed and lightweight arrival tensors for each Lambda.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fixed_csi_benchmark import (
    build_fixed_csi_trace_cache,
    save_trace_cache,
)
from utils.project_setup import load_config, physics_tag


def comma_list(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, required=True)
    parser.add_argument("--num_slots", type=int, default=5000)
    parser.add_argument("--arrival_rates", default="0.5,1,2,4,8")
    parser.add_argument("--mc_seeds", default="42,43,44")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CSI_MODE": "perfect",
    })
    arrival_rates = comma_list(args.arrival_rates, float)
    mc_seeds = comma_list(args.mc_seeds, int)
    output = args.output or (
        "datasets/queue_traces/"
        f"fixed_numUAV={args.num_uav}_{physics_tag(config)}_"
        f"slots={args.num_slots}_mc={len(mc_seeds)}.pt"
    )
    payload = build_fixed_csi_trace_cache(
        config, args.num_slots, arrival_rates, mc_seeds
    )
    print(save_trace_cache(payload, output))


if __name__ == "__main__":
    main()
