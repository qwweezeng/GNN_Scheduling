#!/usr/bin/env python3
"""Paper experiment: one-slot max-weight utility on paired CSI/queue states.

This experiment isolates decision quality from the queue trajectories that each
policy creates.  Every policy sees exactly the same channel and queue snapshot.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.queue_schedulers import (  # noqa: E402
    ConferenceGATNonemptyScheduler,
    RandomNonemptyScheduler,
    SNRSingleLinkScheduler,
    centralized_mws_greedy,
)
from models.local_physical_reuse import (  # noqa: E402
    GATExpectedRateLocalPhysicalReuseScheduler,
    PurePhysicalReuseScheduler,
)
from utils.project_setup import (  # noqa: E402
    build_network_generator,
    load_channel_gat,
    load_config,
    load_end_to_end_queue_gat,
)
from utils.queue_features import relative_queue_feature  # noqa: E402
from utils.scheduling_objective import mws_objective  # noqa: E402


def end_to_end_action(model, state, queues):
    device = next(model.parameters()).device
    node_features = torch.cat([
        state.vertex_features[:, :1].to(device),
        relative_queue_feature(queues).to(device).unsqueeze(1),
    ], dim=1)
    with torch.no_grad():
        probabilities, _ = model(
            node_features,
            state.edge_index.to(device),
            state.edge_weights.to(device),
        )
    action = probabilities.reshape(-1) > float(model.decision_threshold)
    action &= torch.as_tensor(queues, device=device).reshape(-1) > 0
    return action.float().cpu().unsqueeze(1)


def queue_snapshot(num_links, seed):
    """Heterogeneous nonnegative queues with reproducible bursty hotspots."""
    rng = np.random.default_rng(int(seed))
    queues = rng.poisson(2.0, size=num_links).astype(np.float32)
    count = max(1, int(np.ceil(0.2 * num_links)))
    hotspots = rng.choice(num_links, size=count, replace=False)
    queues[hotspots] += rng.poisson(8.0, size=count).astype(np.float32)
    if not np.any(queues > 0):
        queues[rng.integers(0, num_links)] = 1.0
    return torch.from_numpy(queues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp_example.yaml")
    parser.add_argument("--num_uav", type=int, default=10)
    parser.add_argument("--num_snapshots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel_model_num_uav", type=int, default=10)
    parser.add_argument(
        "--queue_gat_checkpoint",
        default=(
            "results/models/"
            "gat_queue_mws_v3_numUAV=10_tx=10dBm_csi=perfect.pth"
        ),
    )
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CSI_MODE": "perfect",
    })
    generator = build_network_generator(config)
    channel_model, channel_path = load_channel_gat(
        config, args.channel_model_num_uav, device="cpu"
    )
    queue_model, queue_path = None, None
    checkpoint = Path(args.queue_gat_checkpoint)
    if args.queue_gat_checkpoint and checkpoint.exists():
        queue_model, queue_path = load_end_to_end_queue_gat(
            config, checkpoint, device="cpu"
        )

    schedulers = {
        "LocalPhysicalReuse-v2": (
            GATExpectedRateLocalPhysicalReuseScheduler(channel_model), None
        ),
        "PurePhysicalReuse": (PurePhysicalReuseScheduler(channel_model), None),
        "Conference-GAT-nonempty": (
            ConferenceGATNonemptyScheduler(channel_model), None
        ),
        "Random-nonempty": (RandomNonemptyScheduler(seed=args.seed + 99), None),
        "SNR-SingleLink": (SNRSingleLinkScheduler(), None),
    }
    if queue_model is not None:
        schedulers["EndToEnd-GAT-q"] = (None, queue_model)

    records = {name: [] for name in schedulers}
    active_counts = {name: [] for name in schedulers}
    reference_values = []
    for offset in range(args.num_snapshots):
        torch.manual_seed(args.seed + offset)
        np.random.seed(args.seed + offset)
        state = generator.generate_network(
            csi_mode="perfect", include_queue_feature=False
        )
        queues = queue_snapshot(state.num_vertices, args.seed + 100_003 + offset)
        reference_action = centralized_mws_greedy(state, queues)
        reference = mws_objective(reference_action, queues, state)
        reference_values.append(reference)
        for name, (scheduler, model) in schedulers.items():
            action = (
                scheduler.schedule_with_state(state, queues)
                if scheduler is not None
                else end_to_end_action(model, state, queues)
            )
            utility = mws_objective(action, queues, state)
            records[name].append(utility / reference if reference > 0 else 1.0)
            active_counts[name].append(float(action.sum()))

    summary = {}
    for name in schedulers:
        ratios = np.asarray(records[name], dtype=float)
        counts = np.asarray(active_counts[name], dtype=float)
        summary[name] = {
            "mean_utility_ratio_to_mws_greedy": float(ratios.mean()),
            "utility_ratio_ci95": float(
                1.96 * ratios.std(ddof=1) / np.sqrt(ratios.size)
            ) if ratios.size > 1 else 0.0,
            "median_utility_ratio_to_mws_greedy": float(np.median(ratios)),
            "mean_active_links": float(counts.mean()),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or (
        f"results/queue/fixed_csi/{timestamp}_snapshot_N{args.num_uav}"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "fixed_csi_snapshot_utility",
        "setting": {
            "num_uav": args.num_uav,
            "num_snapshots": args.num_snapshots,
            "tx_power_dbm": 10.0,
            "csi_mode": "perfect",
            "queue_distribution": "Poisson(2) plus 20% bursty hotspots",
        },
        "channel_checkpoint": channel_path,
        "queue_checkpoint": queue_path,
        "reference_note": (
            "Centralized-MWS-Greedy is a full-information reference, not an "
            "exact optimum or mathematical upper bound."
        ),
        "mean_reference_utility": float(np.mean(reference_values)),
        "summary": summary,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
