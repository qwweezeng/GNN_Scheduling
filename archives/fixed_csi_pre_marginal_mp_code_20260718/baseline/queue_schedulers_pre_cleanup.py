"""Reference schedulers used by the fixed-CSI queue benchmark.

The third-party source trees under :mod:`baseline` remain untouched.  This
module only contains small adapters and in-project reference algorithms.
"""

import torch

from utils.scheduling_objective import mws_objective


class ConferenceGATNonemptyScheduler:
    """Conference channel-only GAT with only an empty-queue feasibility mask.

    Queue magnitudes are never exposed to the GAT.  The nonempty mask avoids
    wasting a transmission on a link that has no packet and is stated
    explicitly in benchmark labels.
    """

    def __init__(self, channel_model):
        self.channel_model = channel_model
        self._action_cache = {}

    def eval(self):
        self.channel_model.eval()
        return self

    def schedule_with_state(self, state, queues):
        device = next(self.channel_model.parameters()).device
        key = (id(state), str(device))
        cached = self._action_cache.get(key)
        action = cached[1] if cached is not None and cached[0] is state else None
        if action is None:
            with torch.no_grad():
                probabilities, _ = self.channel_model(
                    state.vertex_features[:, :1].to(device),
                    state.edge_index.to(device),
                    state.edge_weights.to(device),
                )
            threshold = float(
                getattr(self.channel_model, "decision_threshold", 0.5)
            )
            action = probabilities.reshape(-1) > threshold
            self._action_cache[key] = (state, action)
        nonempty = torch.as_tensor(queues, device=device).reshape(-1) > 0
        return (action & nonempty).to(torch.float32).unsqueeze(1)


class RandomNonemptyScheduler:
    """Random lower bound that never activates an empty queue."""

    def __init__(self, activation_probability=0.3, seed=12345):
        self.activation_probability = float(activation_probability)
        self.generator = torch.Generator().manual_seed(int(seed))

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        del state
        queues = torch.as_tensor(queues).reshape(-1).cpu()
        action = (
            torch.rand(queues.numel(), generator=self.generator)
            < self.activation_probability
        )
        return (action & (queues > 0)).to(torch.float32).unsqueeze(1)


class SNRSingleLinkScheduler:
    """Activate only the highest-SNR nonempty link.

    This is deliberately named ``SNR-SingleLink`` rather than ``Greedy``;
    it is a conservative single-link sanity baseline, not MWS greedy.
    """

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        queues = torch.as_tensor(queues).reshape(-1).cpu()
        action = torch.zeros_like(queues)
        nonempty = (queues > 0).nonzero(as_tuple=True)[0]
        if nonempty.numel() > 0:
            snr = state.vertex_features[:, 0].detach().cpu()
            winner = nonempty[torch.argmax(snr[nonempty])]
            action[winner] = 1.0
        return action.unsqueeze(1)


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


def _link_gain_matrix(state):
    """Return G[source link, affected receiver link] with zero diagonal."""
    mode = getattr(state, "csi_mode", "perfect")
    if mode == "large_scale":
        signal_power = getattr(state, "signal_power_large_scale", None)
    elif mode == "quantized":
        signal_power = getattr(state, "signal_power_quant", None)
    else:
        signal_power = state.signal_power
    if signal_power is None:
        signal_power = state.signal_power
    links = state.links.detach().cpu()
    gain = signal_power.detach().cpu()[
        links[:, 0, None], links[None, :, 1]
    ].clone()
    gain.fill_diagonal_(0.0)
    direct = signal_power.detach().cpu()[links[:, 0], links[:, 1]]
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
