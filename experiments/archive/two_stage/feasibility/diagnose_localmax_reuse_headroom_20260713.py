#!/usr/bin/env python3
"""Measure the ideal headroom of LocalMax plus exact positive reuse joins."""

import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.queue_model_diagnostics import load_channel_model
from experiments.run_parallel_mpnn_queue_experiment import (
    QueueAwareRandomScheduler, jsonify,
)
from experiments.run_queue_experiment import (
    CentralizedMWSGreedyScheduler,
    _load_or_generate_dataset,
    build_generator,
    load_config,
    simulation_phase,
)
from experiments.two_stage_v1_diagnostics import build_stratified_samples
from models.two_stage import QueueChannelLocalMaxScheduler
from utils.mws import mws_objective


class ExactLocalMaxReuseScheduler:
    """Diagnostic oracle: LocalMax base followed by exact positive joins."""

    def __init__(self, localmax):
        self.localmax = localmax

    def eval(self):
        self.localmax.eval()
        return self

    def schedule_with_state(self, state, queues):
        action = self.localmax.schedule_with_state(
            state, queues
        ).reshape(-1).cpu()
        current = float(mws_objective(action, queues.cpu(), state))
        while True:
            candidates = (
                (queues.cpu() > 0) & (action <= 0)
            ).nonzero(as_tuple=True)[0]
            if candidates.numel() == 0:
                break
            utilities = []
            for candidate in candidates:
                proposal = action.clone()
                proposal[candidate] = 1.0
                utilities.append(mws_objective(
                    proposal, queues.cpu(), state
                ))
            utilities = torch.as_tensor(utilities)
            best = int(torch.argmax(utilities))
            if float(utilities[best]) <= current + 1e-9:
                break
            action[candidates[best]] = 1.0
            current = float(utilities[best])
        return action.unsqueeze(1)


class ExactParallelReuseScheduler:
    """One parallel join round using exact individual marginal utilities."""

    def __init__(self, localmax):
        self.localmax = localmax

    def eval(self):
        self.localmax.eval()
        return self

    def schedule_with_state(self, state, queues):
        base = self.localmax.schedule_with_state(
            state, queues
        ).reshape(-1).cpu()
        current = float(mws_objective(base, queues.cpu(), state))
        margins = torch.full_like(base, float("-inf"))
        inactive = ((queues.cpu() > 0) & (base <= 0)).nonzero(
            as_tuple=True
        )[0]
        for candidate in inactive:
            proposal = base.clone()
            proposal[candidate] = 1.0
            margins[candidate] = (
                mws_objective(proposal, queues.cpu(), state) - current
            )
        joiners = margins > 1e-9
        if state.edge_index.numel() > 0:
            source, target = state.edge_index.cpu()
            for u, v in zip(source.tolist(), target.tolist()):
                if not (joiners[u] and joiners[v]):
                    continue
                if margins[u] < margins[v] or (
                    margins[u] == margins[v] and u > v
                ):
                    joiners[u] = False
                else:
                    joiners[v] = False
        return torch.maximum(base, joiners.float()).unsqueeze(1)


def main():
    cfg = load_config("configs/exp_example.yaml")
    cfg.update({
        "NUM_UAV": 10, "TX_POWER": 10.0, "CSI_MODE": "perfect",
        "TEST_SIZE": 20,
    })
    channel_model, _ = load_channel_model(cfg, 10, torch.device("cpu"))
    localmax = QueueChannelLocalMaxScheduler(channel_model)
    oracle_reuse = ExactLocalMaxReuseScheduler(localmax)
    parallel_reuse = ExactParallelReuseScheduler(localmax)

    states = _load_or_generate_dataset(
        build_generator(cfg), cfg, 10, False, "test", (0, 20)
    )
    grouped = build_stratified_samples(states, (0, 20), 44)
    totals = {
        "local_utility": 0.0, "reuse_utility": 0.0,
        "teacher_utility": 0.0, "local_active": 0.0,
        "reuse_active": 0.0, "count": 0,
        "parallel_utility": 0.0, "parallel_active": 0.0,
    }
    with torch.no_grad():
        for samples in grouped.values():
            for state, queues, teacher in samples:
                local_action = localmax.schedule_with_state(
                    state, queues
                ).reshape(-1)
                reuse_action = oracle_reuse.schedule_with_state(
                    state, queues
                ).reshape(-1)
                parallel_action = parallel_reuse.schedule_with_state(
                    state, queues
                ).reshape(-1)
                totals["local_utility"] += mws_objective(
                    local_action, queues, state
                )
                totals["reuse_utility"] += mws_objective(
                    reuse_action, queues, state
                )
                totals["teacher_utility"] += mws_objective(
                    teacher.reshape(-1), queues, state
                )
                totals["parallel_utility"] += mws_objective(
                    parallel_action, queues, state
                )
                totals["local_active"] += float(local_action.sum())
                totals["reuse_active"] += float(reuse_action.sum())
                totals["parallel_active"] += float(parallel_action.sum())
                totals["count"] += 1

    static = {
        "localmax_mws_ratio": (
            totals["local_utility"] / totals["teacher_utility"]
        ),
        "oracle_reuse_mws_ratio": (
            totals["reuse_utility"] / totals["teacher_utility"]
        ),
        "parallel_reuse_mws_ratio": (
            totals["parallel_utility"] / totals["teacher_utility"]
        ),
        "localmax_average_active": (
            totals["local_active"] / totals["count"]
        ),
        "oracle_reuse_average_active": (
            totals["reuse_active"] / totals["count"]
        ),
        "parallel_reuse_average_active": (
            totals["parallel_active"] / totals["count"]
        ),
    }
    models = {
        "Random-Q": (QueueAwareRandomScheduler(), "Queue-aware Random p=0.3"),
        "LocalMax": (localmax, "One-hop LocalMax"),
        "LocalMax+ExactReuse": (
            oracle_reuse, "Diagnostic exact positive reuse joins"
        ),
        "LocalMax+ParallelReuse": (
            parallel_reuse, "Diagnostic one-round parallel exact reuse"
        ),
        "MWS-Greedy": (CentralizedMWSGreedyScheduler(), "MWS-Greedy"),
    }
    dynamic, _ = simulation_phase(
        build_generator(cfg), models, cfg,
        SimpleNamespace(
            num_slots=500, arrival_rates=[2.0], mc_seeds=[42],
            slots_per_block=None,
        ),
    )
    report = {
        "physical_setting": "10 UAV / 10 dBm / perfect CSI",
        "scope": (
            "Diagnostic upper bound only; exact reuse joins use centralized "
            "full-utility evaluation and are not deployable."
        ),
        "static": static,
        "dynamic": dynamic,
    }
    os.makedirs("results/diagnostics", exist_ok=True)
    path = (
        "results/diagnostics/localmax_exact_reuse_headroom_tx10_"
        + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    with open(path, "w", encoding="utf-8") as file:
        json.dump(jsonify(report), file, indent=2)
    print(json.dumps({"report": path, "static": static}, indent=2))


if __name__ == "__main__":
    main()
