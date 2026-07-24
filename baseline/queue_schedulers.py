"""Reference schedulers with the common paper queue interface.

The third-party source trees under :mod:`baseline` remain untouched.  This
module only contains small adapters and in-project reference algorithms.
"""

import torch
import numpy as np

from baseline.fplinq_numpy import binarize_fplinq, optimize_fplinq


class RandomNonemptyScheduler:
    """Activate each nonempty link independently with fixed probability."""

    model_version = "random_nonempty_v1"

    def __init__(self, activation_probability=0.05, seed=0):
        self.activation_probability = float(activation_probability)
        self.generator = torch.Generator().manual_seed(int(seed))

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        queues = torch.as_tensor(queues).reshape(-1).cpu()
        draw = torch.rand(queues.numel(), generator=self.generator)
        return (
            (draw < self.activation_probability) & (queues > 0)
        ).float().unsqueeze(1)


class HardConflictLocalGreedyScheduler:
    """Zhao-style distributed LGS adapted to the project conflict graph.

    Vertex utility is ``Q_i C_i^solo``.  In every contention iteration, all
    remaining conflict-local utility maxima win in parallel; winners and every
    hard-conflicting neighbor are then removed.  The returned schedule is a
    maximal independent set of the symmetrized hard conflict graph.
    """

    model_version = "zhao_hard_conflict_lgs_adaptation_v1"

    def __init__(self):
        self.last_diagnostics = {}

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        from models.local_physical_reuse import (
            local_edge_gains,
            select_local_maxima,
        )

        queues = torch.as_tensor(
            queues, dtype=torch.float32
        ).reshape(-1).cpu()
        direct, _, _, _ = local_edge_gains(state)
        noise = float(state.noise_power)
        gap = float(getattr(state, "snr_gap", 1.0))
        solo_rate = torch.log2(1.0 + direct / noise / gap)
        utility = queues * solo_rate
        remaining = queues > 0
        scheduled = torch.zeros_like(remaining)
        source, target = state.edge_index.detach().cpu()
        iterations = 0
        messages = 0
        while remaining.any():
            round_utility = torch.where(
                remaining,
                utility,
                torch.full_like(utility, -float("inf")),
            )
            winners = select_local_maxima(
                round_utility, remaining, state.edge_index
            ).cpu()
            if not winners.any():
                raise RuntimeError("LGS made no progress on a nonempty graph")
            scheduled |= winners
            blocked = winners.clone()
            if source.numel() > 0:
                source_wins = winners[source]
                target_wins = winners[target]
                blocked[target[source_wins]] = True
                blocked[source[target_wins]] = True
                live_edges = remaining[source] & remaining[target]
                messages += int(live_edges.sum())
            remaining &= ~blocked
            iterations += 1
        self.last_diagnostics = {
            "contention_iterations": iterations,
            "edge_message_proxy": messages,
            "active_links": int(scheduled.sum()),
        }
        return scheduled.float().unsqueeze(1)


def _observed_signal_power(state):
    """Return the power matrix visible to the scheduler on CPU."""
    mode = getattr(state, "csi_mode", "perfect")
    if mode == "large_scale":
        signal_power = getattr(state, "signal_power_large_scale", None)
    elif mode == "quantized":
        signal_power = getattr(state, "signal_power_quant", None)
    else:
        signal_power = state.signal_power
    if signal_power is None:
        signal_power = state.signal_power
    return signal_power.detach().cpu().float()


def _link_gain_matrix(state):
    """Return G[source link, affected receiver link] with zero diagonal."""
    signal_power = _observed_signal_power(state)
    links = state.links.detach().cpu()
    gain = signal_power[
        links[:, 0, None], links[None, :, 1]
    ].clone()
    gain.fill_diagonal_(0.0)
    direct = signal_power[links[:, 0], links[:, 1]]
    return gain.float(), direct.float()


def centralized_mws_greedy(state, queues, gain_and_direct=None):
    """Vectorized global-CSI add/delete greedy max-weight reference.

    At each step it evaluates the exact network-wide marginal utility of adding
    every inactive link in parallel, applies the best positive change, and then
    permits improving deletions.  It is a strong centralized reference, but it
    remains a greedy local optimum rather than a mathematical upper bound.
    """
    queues = torch.as_tensor(
        queues, dtype=torch.float32
    ).reshape(-1).cpu().clamp_min(0.0)
    active = torch.zeros(state.num_vertices, dtype=torch.bool)
    if not (queues > 0).any():
        return active.float().unsqueeze(1)

    gain, direct = gain_and_direct or _link_gain_matrix(state)
    noise = float(state.noise_power)
    gap = float(getattr(state, "snr_gap", 1.0))
    interference = torch.zeros_like(direct)
    eps = 1e-9

    while True:
        rates = torch.log2(1.0 + direct / (interference + noise) / gap)
        joined_rates = torch.log2(
            1.0
            + direct.unsqueeze(0)
            / (interference.unsqueeze(0) + gain + noise)
            / gap
        )
        add_delta = queues * rates + (
            (joined_rates - rates.unsqueeze(0))
            * (queues * active.float()).unsqueeze(0)
        ).sum(dim=1)
        add_delta[active | (queues <= 0)] = -float("inf")
        best_delta, best_index = torch.max(add_delta, dim=0)
        if torch.isfinite(best_delta) and float(best_delta) > eps:
            index = int(best_index)
            active[index] = True
            interference += gain[index]
            continue

        current = float((queues * rates * active.float()).sum())
        active_indices = active.nonzero(as_tuple=True)[0]
        if active_indices.numel() == 0:
            break
        removed_interference = (
            interference.unsqueeze(0) - gain[active_indices]
        ).clamp_min(0.0)
        removed_rates = torch.log2(
            1.0
            + direct.unsqueeze(0)
            / (removed_interference + noise)
            / gap
        )
        masks = active.float().unsqueeze(0).repeat(
            active_indices.numel(), 1
        )
        masks[torch.arange(active_indices.numel()), active_indices] = 0.0
        remove_utilities = (removed_rates * queues * masks).sum(dim=1)
        best_utility, position = torch.max(remove_utilities, dim=0)
        if float(best_utility) > current + eps:
            index = int(active_indices[int(position)])
            active[index] = False
            interference -= gain[index]
            continue
        break

    return active.float().unsqueeze(1)


class CentralizedMWSGreedyScheduler:
    """Scheduler adapter for :func:`centralized_mws_greedy`."""

    def __init__(self):
        self._gain_cache = {}

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        cached = self._gain_cache.get(id(state))
        gains = cached[1] if cached is not None and cached[0] is state else None
        if gains is None:
            gains = _link_gain_matrix(state)
            self._gain_cache[id(state)] = (state, gains)
        return centralized_mws_greedy(state, queues, gains)


class WeightedFPLinQScheduler:
    """Queue-weighted FPLinQ reference evaluated on the project channel.

    FPLinQ is centralized and iterative in this adapter.  Missing third-party
    dependencies are reported as an error; paper experiments never substitute
    a different scheduler silently.
    """

    model_version = "queue_weighted_fplinq_received_power_v2"

    def __init__(self, iterations=100, amplitude_threshold=0.5):
        self.iterations = int(iterations)
        self.amplitude_threshold = float(amplitude_threshold)
        self.last_diagnostics = {}

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        if not np.isclose(float(getattr(state, "snr_gap", 1.0)), 1.0):
            raise ValueError(
                "the retained FPLinQ update is validated only for the paper's "
                "0 dB SNR gap"
            )
        signal_power = _observed_signal_power(state)
        links = state.links.detach().cpu()
        # Matrix orientation: [affected receiver link, transmitting link].
        gains = signal_power[
            links[None, :, 0], links[:, None, 1]
        ]
        weights = torch.as_tensor(
            queues, dtype=torch.float32
        ).reshape(-1).clamp_min(0.0).cpu().numpy()
        allocation, diagnostics = optimize_fplinq(
            gains.detach().cpu().numpy(),
            float(state.noise_power),
            weights,
            iterations=self.iterations,
            return_diagnostics=True,
        )
        action = binarize_fplinq(
            allocation, amplitude_threshold=self.amplitude_threshold
        )
        action *= (weights > 0).astype(np.float32)
        self.last_diagnostics = {
            "iterations": diagnostics.iterations,
            "max_last_update": diagnostics.max_last_update,
            "continuous_nonzero_links": int(np.count_nonzero(allocation)),
            "active_links": int(np.count_nonzero(action)),
            "global_csi": True,
            "matrix_semantics": "received_power_at_rx_i_from_tx_j",
        }
        return torch.from_numpy(action).unsqueeze(1)


__all__ = [
    "CentralizedMWSGreedyScheduler",
    "HardConflictLocalGreedyScheduler",
    "RandomNonemptyScheduler",
    "WeightedFPLinQScheduler",
    "centralized_mws_greedy",
]
