#!/usr/bin/env python3
"""Same-physics snapshot comparison of MarginalMP and hard-conflict LGS."""

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
    CentralizedMWSGreedyScheduler,
    HardConflictLocalGreedyScheduler,
)
from models.initial_schedule_construction import (  # noqa: E402
    DirectLinkJoinScheduler,
    DirectLinkLocalMaxScheduler,
)
from models.marginal_mp import DirectLinkMarginalMP  # noqa: E402
from utils.project_setup import build_network_generator, load_config  # noqa: E402
from utils.queue_snapshots import queue_snapshot  # noqa: E402
from utils.scheduling_objective import mws_objective  # noqa: E402


REFERENCE = "direct_link_join"
CENTRAL = "centralized_greedy"
PROPOSED = "direct_link_marginal_mp"
LGS = "hard_conflict_lgs"
LOCALMAX = "direct_link_localmax"


def summarize(records, method):
    ratios_reference = np.asarray([
        row[method] / max(abs(row[REFERENCE]), 1e-12) for row in records
    ])
    ratios_central = np.asarray([
        row[method] / max(abs(row[CENTRAL]), 1e-12) for row in records
    ])
    return {
        "num_snapshots": len(records),
        "mean_ratio_vs_direct_link_join": float(ratios_reference.mean()),
        "improvement_ci95_vs_direct_link_join": float(
            1.96
            * (ratios_reference - 1.0).std(ddof=1)
            / np.sqrt(len(records))
        ) if len(records) > 1 else 0.0,
        "strict_win_fraction_vs_direct_link_join": float(
            np.mean(ratios_reference > 1.0 + 1e-9)
        ),
        "strict_loss_fraction_vs_direct_link_join": float(
            np.mean(ratios_reference < 1.0 - 1e-9)
        ),
        "mean_ratio_vs_centralized_greedy": float(ratios_central.mean()),
        "mean_active_links": float(np.mean([
            row[f"{method}_active"] for row in records
        ])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--num_graphs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--output",
        default="results/paper/fixed_csi_snapshot/marginal_mp_vs_lgs_N20.json",
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
    schedulers = {
        LGS: HardConflictLocalGreedyScheduler(),
        LOCALMAX: DirectLinkLocalMaxScheduler(),
        REFERENCE: DirectLinkJoinScheduler(),
        PROPOSED: DirectLinkMarginalMP(rounds=args.rounds),
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
                "num_directed_conflict_edges": int(state.edge_index.size(1)),
            }
            for name, scheduler in schedulers.items():
                action = scheduler.schedule_with_state(state, queues)
                row[name] = float(mws_objective(action, queues, state))
                row[f"{name}_active"] = int(action.sum())
            row["lgs_contention_iterations"] = schedulers[
                LGS
            ].last_diagnostics["contention_iterations"]
            row["marginal_mp_selected_per_round"] = schedulers[
                PROPOSED
            ].last_diagnostics["selected_toggles_per_round"]
            row["marginal_mp_logical_exchange_rounds"] = schedulers[
                PROPOSED
            ].last_diagnostics["logical_neighbor_exchange_rounds"]
            row["marginal_mp_physical_externality_pairs_per_round"] = schedulers[
                PROPOSED
            ].last_diagnostics["physical_externality_pairs_per_round"]
            row["marginal_mp_arbitration_edge_messages_per_round"] = schedulers[
                PROPOSED
            ].last_diagnostics["arbitration_directed_edge_messages_per_round"]
            records.append(row)
            print(
                f"seed={graph_seed} profile={profile:<13} "
                f"LGS/Join={row[LGS]/row[REFERENCE]:.3f} "
                f"MarginalMP/Join={row[PROPOSED]/row[REFERENCE]:.3f} "
                f"MarginalMP/Central={row[PROPOSED]/row[CENTRAL]:.3f}"
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
    proposed = overall[PROPOSED]
    lgs = overall[LGS]
    payload = {
        "experiment": "marginal_mp_vs_hard_conflict_lgs_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            **vars(args),
            "marginal_mp_rounds": args.rounds,
            "marginal_mp_initializer": "Direct-Link LocalMax (no Join)",
        },
        "method_display_names": {
            LGS: "Hard-Conflict LGS",
            LOCALMAX: "Direct-Link LocalMax",
            REFERENCE: "Direct-Link Join",
            PROPOSED: "Direct-Link MarginalMP",
            CENTRAL: "Centralized Greedy",
        },
        "fairness_definition": {
            "lgs_graph": (
                "Symmetrized project conflict graph treated as a hard "
                "exclusion graph."
            ),
            "lgs_vertex_utility": (
                "Queue length times interference-free direct-link quality."
            ),
            "all_final_actions_evaluated_by": (
                "The same physical-SINR queue-weighted utility."
            ),
            "warning": (
                "This is a same-scenario adaptation of Zhao's downstream "
                "LGS, not a reproduction of Zhao's ER-graph experiment."
            ),
        },
        "decision": (
            "PROCEED_FIXED_CSI_MARGINAL_MP"
            if proposed["mean_ratio_vs_direct_link_join"] >= 1.02
            and proposed["strict_loss_fraction_vs_direct_link_join"] <= 0.20
            and proposed["mean_ratio_vs_direct_link_join"]
            > lgs["mean_ratio_vs_direct_link_join"]
            else "STOP_OR_REDESIGN_MARGINAL_MP"
        ),
        "overall": overall,
        "by_profile": by_profile,
        "mean_lgs_contention_iterations": float(np.mean([
            row["lgs_contention_iterations"] for row in records
        ])),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "decision": payload["decision"],
        "overall": overall,
        "mean_lgs_contention_iterations": (
            payload["mean_lgs_contention_iterations"]
        ),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
