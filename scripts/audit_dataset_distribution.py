#!/usr/bin/env python3
"""One-off audit for legacy channel-dataset physical provenance."""

import json
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.project_setup import build_network_generator, load_config


def summarize(states):
    nodes = np.asarray([state.num_vertices for state in states], dtype=float)
    edges = np.asarray([
        state.edge_index.size(1) for state in states
    ], dtype=float)
    snr = torch.cat([state.vertex_features[:, 0] for state in states]).numpy()
    densities = edges / np.maximum(nodes * (nodes - 1), 1)
    return {
        "num_graphs": len(states),
        "links_mean": float(nodes.mean()),
        "links_std": float(nodes.std()),
        "links_min": int(nodes.min()),
        "links_max": int(nodes.max()),
        "snr_db_mean": float(snr.mean()),
        "snr_db_std": float(snr.std()),
        "directed_edge_density_mean": float(densities.mean()),
    }


def fresh_states(config, tx_power, count, seed):
    local = dict(config)
    local["NUM_UAV"] = 10
    local["TX_POWER"] = tx_power
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = build_network_generator(local)
    return [
        generator.generate_network(csi_mode=local.get("CSI_MODE", "perfect"))
        for _ in range(count)
    ]


def main():
    config = load_config("configs/paper.yaml")
    with open("datasets/dataset_numUAV=10.pkl", "rb") as file:
        cached = pickle.load(file)
    report = {
        "cache_path": "datasets/dataset_numUAV=10.pkl",
        "cache_key_limitation": (
            "filename contains num_uav only; physical metadata is not "
            "validated on legacy cache load"
        ),
        "cached_train_first_200": summarize(cached["train_set"][:200]),
        "cached_test_first_32": summarize(cached["test_set"][:32]),
        "fresh_tx10_32": summarize(fresh_states(config, 10.0, 32, 42)),
        "fresh_tx22_32": summarize(fresh_states(config, 22.0, 32, 42)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
