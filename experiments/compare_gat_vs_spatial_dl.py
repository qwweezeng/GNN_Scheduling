#!/usr/bin/env python3
"""Paper Fig. 1: full-buffer Channel-GAT versus retained baselines.

The script reads the canonical Rician test split and a metadata-checked GAT
checkpoint.  Every returned schedule is settled by the project's Shannon-rate
function.  Missing third-party baselines fail loudly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model import calculate_achievable_rate  # noqa: E402
from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.baseline import (  # noqa: E402
    eval_fplinq,
    eval_local_greedy,
    eval_spatial_dl,
    random_activate,
    simple_greedy,
)
from utils.project_setup import (  # noqa: E402
    load_channel_gat,
    load_channel_gat_checkpoint,
    load_config,
)


def _canonical_test(config, num_uav, test_size):
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 500, "val": 128, "test": 128}
    expected = expected_dataset_metadata(config, num_uav, seeds, sizes)
    _, test_path = paper_dataset_paths(Path("datasets/paper"), num_uav, seeds, sizes)
    payload = load_validated_dataset(test_path, expected, ("test",))
    if int(test_size) > len(payload["test_set"]):
        raise ValueError(
            f"Requested {test_size} samples, canonical test split has "
            f"{len(payload['test_set'])}"
        )
    return payload["test_set"][: int(test_size)], test_path, expected["fingerprint"]


def _settle(actions, states):
    utilities, fractions = [], []
    for action, state in zip(actions, states):
        action = torch.as_tensor(action, dtype=torch.float32).reshape(-1, 1)
        utilities.append(float(calculate_achievable_rate(
            action, state, binary=True, detach_hard=True
        ).sum()))
        fractions.append(float(action.mean()))
    return utilities, fractions


def _gat_actions(model, states):
    result = []
    model.eval()
    with torch.no_grad():
        for state in states:
            probability, _ = model(
                state.vertex_features[:, :1], state.edge_index, state.edge_weights
            )
            result.append((probability >= 0.5).float().cpu())
    return result


def _summary(values, active):
    array = np.asarray(values, dtype=float)
    active_array = np.asarray(active, dtype=float)
    return {
        "mean_sum_spectral_efficiency": float(array.mean()),
        "std_sum_spectral_efficiency": float(array.std(ddof=0)),
        "standard_error": float(array.std(ddof=1) / np.sqrt(len(array)))
        if len(array) > 1 else 0.0,
        "mean_active_fraction": (
            float(np.nanmean(active_array))
            if np.isfinite(active_array).any() else None
        ),
        "per_sample": array.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--gat_model", default="")
    parser.add_argument(
        "--methods",
        default="Random,Top-SNR,Local Greedy,Spatial DL,FPLinQ,Channel-GAT",
    )
    parser.add_argument(
        "--output", default="results/paper/full_buffer/benchmark.json"
    )
    args = parser.parse_args()

    random.seed(10088)
    np.random.seed(10088)
    torch.manual_seed(10088)
    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    states, dataset_path, fingerprint = _canonical_test(
        config, args.num_uav, args.num_samples
    )
    requested = [name.strip() for name in args.methods.split(",") if name.strip()]
    results = {}

    for name in requested:
        if name == "Random":
            torch.manual_seed(20260719)
            actions = [random_activate(state, 0.05) for state in states]
            values, fractions = _settle(actions, states)
        elif name == "Top-SNR":
            actions = [simple_greedy(state, 0.10) for state in states]
            values, fractions = _settle(actions, states)
        elif name == "Local Greedy":
            values = eval_local_greedy(states, interference_threshold=0.3)
            fractions = [float("nan")] * len(values)
        elif name == "Spatial DL":
            values = eval_spatial_dl(states)
            fractions = [float("nan")] * len(values)
        elif name == "FPLinQ":
            values = eval_fplinq(states, use_fading=True)
            fractions = [float("nan")] * len(values)
        elif name == "Channel-GAT":
            if args.gat_model:
                model, checkpoint_path = load_channel_gat_checkpoint(
                    config, args.gat_model, args.num_uav, device="cpu"
                )
            else:
                model, checkpoint_path = load_channel_gat(
                    config, args.num_uav, device="cpu"
                )
            actions = _gat_actions(model, states)
            values, fractions = _settle(actions, states)
            results["channel_gat_checkpoint"] = checkpoint_path
        else:
            raise ValueError(f"Unknown method: {name}")
        results[name] = _summary(values, fractions)
        print(f"{name}: {results[name]['mean_sum_spectral_efficiency']:.4f}")

    output = Path(args.output)
    if output.name == "benchmark.json" and args.num_uav != 20:
        output = output.with_name(f"benchmark_N{args.num_uav}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "paper_full_buffer_benchmark_v1",
        "definition": "all actions settled on the canonical perfect Rician channel",
        "num_uav": args.num_uav,
        "num_samples": len(states),
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": fingerprint,
        "config": config,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
