#!/usr/bin/env python3
"""Generate the paired same-(Q,H) source data for TNSE Fig. 3.

Every method sees the identical Rician state and queue snapshot.  Candidate
actions are retained, then all are subjected to the same single-radio /
half-duplex matching projection before the true-channel MWS objective is
settled.  This is intentionally a decision-quality experiment, not a queue
stability simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baseline.queue_schedulers import (  # noqa: E402
    CentralizedMWSGreedyScheduler, HardConflictLocalGreedyScheduler,
)
from models.initial_schedule_construction import (  # noqa: E402
    DirectLinkJoinScheduler, DirectLinkLocalMaxScheduler,
    GATAssistedLocalMaxScheduler,
)
from models.marginal_mp import (  # noqa: E402
    DirectLinkMarginalMP, GATAssistedMarginalMP,
)
from utils.feasibility import (  # noqa: E402
    project_shared_uav_feasible, shared_uav_action_audit,
)
from utils.project_setup import (  # noqa: E402
    build_network_generator, load_channel_gat, load_config,
)
from utils.queue_snapshots import queue_snapshot  # noqa: E402
from utils.scheduling_objective import mws_objective  # noqa: E402


METHODS = (
    "GAT-Assisted MarginalMP", "Direct-Link MarginalMP",
    "GAT-Assisted LocalMax", "Direct-Link LocalMax", "Direct-Link Join",
    "Hard-Conflict LGS", "Centralized Greedy",
)
PROFILES = ("sparse", "heterogeneous", "saturated")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _paper_config(num_uav: int) -> dict:
    config = load_config(REPO_ROOT / "configs/paper.yaml")
    config.update({
        "NUM_UAV": int(num_uav), "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician", "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    return config


def _direct_priority(state, queues: torch.Tensor) -> torch.Tensor:
    links = state.links.detach().cpu()
    power = state.signal_power.detach().cpu().float()
    direct = power[links[:, 0], links[:, 1]]
    solo_rate = torch.log2(1.0 + direct / float(state.noise_power))
    return torch.as_tensor(queues, dtype=torch.float32).reshape(-1) * solo_rate


def _message_record(name: str, scheduler) -> dict:
    diagnostics = getattr(scheduler, "last_diagnostics", {}) or {}
    if "logical_neighbor_exchange_rounds" in diagnostics:
        logical_rounds = int(diagnostics["logical_neighbor_exchange_rounds"])
    elif name.endswith("LocalMax") or name == "Hard-Conflict LGS":
        logical_rounds = 1
    elif name.endswith("Join"):
        logical_rounds = 2
    else:
        logical_rounds = 0
    return {
        "logical_neighbor_exchange_rounds": logical_rounds,
        "physical_externality_pairs": int(sum(
            diagnostics.get("physical_externality_pairs_per_round", [])
        )),
        "arbitration_directed_edge_messages": int(sum(
            diagnostics.get("arbitration_directed_edge_messages_per_round", [])
        )),
        "lgs_contention_iterations": diagnostics.get("contention_iterations"),
        "lgs_edge_message_proxy": diagnostics.get("edge_message_proxy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-uav", type=int, default=20)
    parser.add_argument("--num-graphs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = _paper_config(args.num_uav)
    channel_model, checkpoint = load_channel_gat(config, args.num_uav, device="cpu")
    generator = build_network_generator(config)
    factories = {
        "GAT-Assisted MarginalMP": lambda: GATAssistedMarginalMP(channel_model, args.rounds),
        "Direct-Link MarginalMP": lambda: DirectLinkMarginalMP(args.rounds),
        "GAT-Assisted LocalMax": lambda: GATAssistedLocalMaxScheduler(channel_model),
        "Direct-Link LocalMax": DirectLinkLocalMaxScheduler,
        "Direct-Link Join": DirectLinkJoinScheduler,
        "Hard-Conflict LGS": HardConflictLocalGreedyScheduler,
        "Centralized Greedy": CentralizedMWSGreedyScheduler,
    }
    records = []
    for graph_index in range(args.num_graphs):
        graph_seed = int(args.seed) + graph_index
        _seed(graph_seed)
        state = generator.generate_network(csi_mode="perfect", include_queue_feature=False)
        for profile_index, profile in enumerate(PROFILES):
            queues = queue_snapshot(
                state.num_vertices,
                graph_seed + 100_003 + 10_000 * profile_index,
                profile,
            )
            for method in METHODS:
                scheduler = factories[method]().eval()
                candidate = scheduler.schedule_with_state(state, queues).reshape(-1).cpu()
                candidate_audit = shared_uav_action_audit(candidate, state.links)
                action, projection = project_shared_uav_feasible(
                    candidate, state.links, priorities=_direct_priority(state, queues)
                )
                final_audit = shared_uav_action_audit(action, state.links)
                if not final_audit["feasible"]:
                    raise RuntimeError(f"{method} produced an infeasible final action")
                records.append({
                    "graph_index": graph_index, "graph_seed": graph_seed,
                    "queue_profile": profile, "method": method,
                    "num_links": int(state.num_vertices),
                    "num_directed_conflict_edges": int(state.edge_index.size(1)),
                    "queue_sum_packets": float(queues.sum()),
                    "mws_utility_queue_weighted_bit_s_hz": float(mws_objective(action, queues, state)),
                    "candidate_active_project_link_indices": torch.nonzero(candidate > 0.5, as_tuple=True)[0].tolist(),
                    "final_active_project_link_indices": torch.nonzero(action.reshape(-1) > 0.5, as_tuple=True)[0].tolist(),
                    "candidate_hard_feasibility": candidate_audit,
                    "projection": projection,
                    "final_hard_feasibility": final_audit,
                    "message_accounting": _message_record(method, scheduler),
                })
                print(
                    f"graph={graph_index:02d} profile={profile:<13} method={method:<27} "
                    f"U={records[-1]['mws_utility_queue_weighted_bit_s_hz']:.3f}", flush=True
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tnse_fig03_same_snapshot_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "settings": {
            "num_uav": args.num_uav, "num_graphs": args.num_graphs,
            "base_seed": args.seed, "marginal_mp_rounds": args.rounds,
            "profiles": list(PROFILES), "methods": list(METHODS),
            "tx_power_dbm": 10.0, "channel_model": "rician",
            "rician_k_db": 5.0, "observed_csi": "perfect instantaneous",
            "service_channel": "current true instantaneous Rician channel",
            "shared_uav_constraint": "single_radio_half_duplex_matching",
            "projection_rule": "common_direct_rate_priority_then_project_link_index",
            "channel_gat_checkpoint": str(checkpoint),
        },
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
