#!/usr/bin/env python3
"""Ablate whether add-only Join is needed before MarginalMP.

This is a structure-selection diagnostic, not a paper baseline collection.
It compares a common Direct-Link LocalMax skeleton followed by either Join,
one to three MarginalMP rounds, or the current Join-plus-two-round pipeline.
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

from baseline.queue_schedulers import CentralizedMWSGreedyScheduler  # noqa: E402
from experiments.diagnose_conditional_negotiation_gate import _UnusedChannelModel  # noqa: E402
from experiments.diagnose_mws_topology_scaling_headroom import queue_snapshot  # noqa: E402
from models.initial_schedule_construction import (  # noqa: E402
    DirectLinkJoinScheduler,
    DirectLinkLocalMaxScheduler,
)
from models.marginal_mp import (  # noqa: E402
    JoinInitializedMarginalMP,
    MarginalMPScheduler,
)
from utils.project_setup import build_network_generator, load_config  # noqa: E402
from utils.scheduling_objective import mws_objective  # noqa: E402


CURRENT = "localmax_join_marginal2"
JOIN = "localmax_join"
CENTRAL = "centralized_greedy"
CANDIDATE2 = "localmax_marginal2"
CANDIDATE3 = "localmax_marginal3"
CANDIDATE4 = "localmax_marginal4"


def _marginal_from_localmax(channel_model, rounds):
    return MarginalMPScheduler(
        DirectLinkLocalMaxScheduler(channel_model),
        rounds=rounds,
        initializer_name="Direct-Link LocalMax",
    )


def summarize(records, method):
    values = np.asarray([row[method] for row in records], dtype=np.float64)
    current = np.asarray([row[CURRENT] for row in records], dtype=np.float64)
    join = np.asarray([row[JOIN] for row in records], dtype=np.float64)
    central = np.asarray([row[CENTRAL] for row in records], dtype=np.float64)
    ratio_current = values / np.maximum(np.abs(current), 1e-12)
    ratio_join = values / np.maximum(np.abs(join), 1e-12)
    ratio_central = values / np.maximum(np.abs(central), 1e-12)
    return {
        "num_snapshots": len(records),
        "mean_ratio_vs_current_join_plus_marginal2": float(
            ratio_current.mean()
        ),
        "worst_ratio_vs_current_join_plus_marginal2": float(
            ratio_current.min()
        ),
        "strict_win_fraction_vs_current": float(
            np.mean(ratio_current > 1.0 + 1e-9)
        ),
        "strict_loss_fraction_vs_current": float(
            np.mean(ratio_current < 1.0 - 1e-9)
        ),
        "mean_ratio_vs_join": float(ratio_join.mean()),
        "mean_ratio_vs_centralized_greedy": float(ratio_central.mean()),
        "mean_active_links": float(np.mean([
            row[f"{method}_active"] for row in records
        ])),
    }


def select_structure(overall):
    """Decide only whether the redundant Join module is needed."""
    four = overall[CANDIDATE4]
    current = overall[CURRENT]
    if (
        four["mean_ratio_vs_join"] >= 1.02
        and four["mean_ratio_vs_centralized_greedy"]
        >= 0.99 * current["mean_ratio_vs_centralized_greedy"]
    ):
        return "REMOVE_JOIN_USE_K_ROUND_MARGINAL_MP"
    return "RETAIN_JOIN_PENDING_FURTHER_DIAGNOSIS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp_example.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--num_graphs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument(
        "--output",
        default=(
            "results/formal/"
            "marginal_mp_join_ablation_N20_rician.json"
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CSI_MODE": "perfect",
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
    })
    generator = build_network_generator(config)
    channel = _UnusedChannelModel()
    schedulers = {
        "localmax": DirectLinkLocalMaxScheduler(channel),
        JOIN: DirectLinkJoinScheduler(channel),
        "localmax_marginal1": _marginal_from_localmax(channel, 1),
        CANDIDATE2: _marginal_from_localmax(channel, 2),
        CANDIDATE3: _marginal_from_localmax(channel, 3),
        CANDIDATE4: _marginal_from_localmax(channel, 4),
        CURRENT: JoinInitializedMarginalMP(channel, rounds=2),
        CENTRAL: CentralizedMWSGreedyScheduler(),
    }
    profiles = ("sparse", "heterogeneous", "saturated")
    records = []
    for graph_index in range(args.num_graphs):
        graph_seed = args.seed + graph_index
        torch.manual_seed(graph_seed)
        np.random.seed(graph_seed)
        state = generator.generate_network(
            csi_mode="perfect", include_queue_feature=False
        )
        for profile_index, profile in enumerate(profiles):
            queues = queue_snapshot(
                state.num_vertices,
                graph_seed + 100_003 + 10_000 * profile_index,
                profile,
            )
            row = {
                "graph_seed": graph_seed,
                "queue_profile": profile,
                "num_links": int(state.num_vertices),
            }
            for name, scheduler in schedulers.items():
                action = scheduler.schedule_with_state(state, queues)
                row[name] = float(mws_objective(action, queues, state))
                row[f"{name}_active"] = int(action.sum())
            records.append(row)
            print(
                f"seed={graph_seed} profile={profile:<13} "
                f"M2/current={row[CANDIDATE2]/row[CURRENT]:.3f} "
                f"M3/current={row[CANDIDATE3]/row[CURRENT]:.3f} "
                f"current/join={row[CURRENT]/row[JOIN]:.3f}"
            )

    methods = tuple(schedulers)
    overall = {method: summarize(records, method) for method in methods}
    by_profile = {
        profile: {
            method: summarize(
                [row for row in records if row["queue_profile"] == profile],
                method,
            ) for method in methods
        } for profile in profiles
    }
    decision = select_structure(overall)
    payload = {
        "experiment": "marginal_mp_join_structure_ablation_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": vars(args),
        "structure_definitions": {
            "localmax": "Direct-Link quality -> LocalMax",
            JOIN: "Direct-Link quality -> LocalMax -> add-only Join",
            "localmax_marginal1": (
                "Direct-Link quality -> LocalMax -> 1 MarginalMP round"
            ),
            CANDIDATE2: (
                "Direct-Link quality -> LocalMax -> 2 MarginalMP rounds"
            ),
            CANDIDATE3: (
                "Direct-Link quality -> LocalMax -> 3 MarginalMP rounds"
            ),
            CANDIDATE4: (
                "Direct-Link quality -> LocalMax -> 4 MarginalMP rounds"
            ),
            CURRENT: (
                "Direct-Link quality -> LocalMax -> Join -> "
                "2 MarginalMP rounds"
            ),
            CENTRAL: "Centralized exact-marginal add/delete greedy reference",
        },
        "selection_rule": {
            "structure_gate": (
                "Remove Join if LocalMax+M4 is at least 2% better than Join "
                "and retains at least 99% of the centralized-greedy ratio "
                "of the legacy Join+M2 pipeline."
            ),
            "round_budget": (
                "K is an explicit interaction budget. Current formal runs "
                "use K=4; K=1..4 are reported only as a cost-utility ablation."
            ),
        },
        "decision": decision,
        "overall": overall,
        "by_profile": by_profile,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "decision": decision,
        "overall": overall,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
