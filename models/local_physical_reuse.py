"""Low-level LocalMax/reuse physics retained for compatibility and CSI work.

This is not the current paper-model entry point.  New fixed-CSI code must use
``models.initial_schedule_construction`` and ``models.marginal_mp``.  The
historical Join implementation remains here only because the retained
Direct-Link Join baseline and several CSI audits share its physical helpers.
"""

from collections import OrderedDict

import torch


_LOCAL_EDGE_GAIN_CACHE = OrderedDict()
_MAX_EDGE_GAIN_CACHE_STATES = 32


def _signal_power_matrix(state):
    """Return received powers consistent with the CSI exposed to scheduling."""
    mode = getattr(state, "csi_mode", "perfect")
    if mode == "large_scale":
        power = getattr(state, "signal_power_large_scale", None)
    elif mode == "quantized":
        power = getattr(state, "signal_power_quant", None)
    else:
        power = state.signal_power
    if power is None:
        power = state.signal_power
    return power.detach().cpu()


def local_edge_gains(state):
    """Return direct and directed conflict-edge gains in O(L+E) storage."""
    key = id(state)
    cached = _LOCAL_EDGE_GAIN_CACHE.get(key)
    if cached is not None and cached[0] is state:
        _LOCAL_EDGE_GAIN_CACHE.move_to_end(key)
        return cached[1], cached[2], cached[3], cached[4]

    signal_power = _signal_power_matrix(state)
    links = state.links.detach().cpu()
    link_tx, link_rx = links[:, 0], links[:, 1]
    direct = signal_power[link_tx, link_rx]
    source, target = state.edge_index.detach().cpu()
    if source.numel() > 0:
        edge_gain = signal_power[link_tx[source], link_rx[target]]
    else:
        edge_gain = torch.empty(0, dtype=direct.dtype)
    _LOCAL_EDGE_GAIN_CACHE[key] = (
        state, direct, source, target, edge_gain
    )
    while len(_LOCAL_EDGE_GAIN_CACHE) > _MAX_EDGE_GAIN_CACHE_STATES:
        _LOCAL_EDGE_GAIN_CACHE.popitem(last=False)
    return direct, source, target, edge_gain


def conflict_graph_local_join_margins(state, queues, base_action):
    """Compute each inactive link's local conditional max-weight gain."""
    queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1).cpu()
    base = torch.as_tensor(
        base_action, dtype=torch.float32
    ).reshape(-1).round().cpu()
    direct, source, target, edge_gain = local_edge_gains(state)
    noise = torch.as_tensor(state.noise_power, dtype=torch.float32).cpu()
    snr_gap = float(getattr(state, "snr_gap", 1.0))

    interference = torch.zeros_like(direct)
    if source.numel() > 0:
        interference.index_add_(0, target, edge_gain * base[source])
    base_efficiency = torch.log2(
        1.0 + direct / (interference + noise) / snr_gap
    )

    existing_change = torch.zeros_like(direct)
    if source.numel() > 0:
        joined_efficiency = torch.log2(
            1.0
            + direct[target]
            / (interference[target] + edge_gain + noise)
            / snr_gap
        )
        edge_change = (
            queues[target]
            * base[target]
            * (joined_efficiency - base_efficiency[target])
        )
        existing_change.index_add_(0, source, edge_change)

    margins = queues * base_efficiency + existing_change
    valid = (queues > 0) & (base <= 0)
    return torch.where(
        valid, margins, torch.full_like(margins, float("-inf"))
    )


def complete_physical_reuse_from_base(state, queues, base_action):
    """Apply the common one-round physics-informed join to a fixed skeleton."""
    base = torch.as_tensor(
        base_action, dtype=torch.float32
    ).reshape(-1).round().cpu()
    margins = conflict_graph_local_join_margins(state, queues, base)
    joiners = select_local_positive_maxima(
        margins, torch.isfinite(margins), state.edge_index
    )
    return torch.maximum(base, joiners.float()).unsqueeze(1), margins, joiners


def select_local_positive_maxima(scores, valid, edge_index):
    """Select positive candidates that beat all conflicting candidates."""
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
    valid = torch.as_tensor(valid, dtype=torch.bool).reshape(-1).cpu()
    candidates = valid & torch.isfinite(scores) & (scores > 0.0)
    if not candidates.any() or edge_index.numel() == 0:
        return candidates

    source, target = edge_index.detach().cpu()
    both = candidates[source] & candidates[target]
    source_loses = both & (
        (scores[source] < scores[target])
        | ((scores[source] == scores[target]) & (source > target))
    )
    target_loses = both & (
        (scores[target] < scores[source])
        | ((scores[target] == scores[source]) & (target > source))
    )
    losers = torch.zeros_like(candidates)
    losers[source[source_loses]] = True
    losers[target[target_loses]] = True
    return candidates & ~losers


def select_local_maxima(scores, valid, edge_index):
    """Select deterministic simultaneous maxima over conflict neighborhoods."""
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    valid = torch.as_tensor(valid, dtype=torch.bool).reshape(-1).to(scores.device)
    winners = valid.clone()
    if not winners.any() or edge_index.numel() == 0:
        return winners
    source, target = edge_index.to(scores.device)
    source_loses = (scores[source] < scores[target]) | (
        (scores[source] == scores[target]) & (source > target)
    )
    target_loses = (scores[target] < scores[source]) | (
        (scores[target] == scores[source]) & (target > source)
    )
    losers = torch.zeros_like(winners)
    losers[source[source_loses]] = True
    losers[target[target_loses]] = True
    return winners & ~losers


class QueueChannelLocalMaxScheduler:
    """One-hop LocalMax skeleton used by the frozen model and its ablations."""

    PRIORITY_MODES = {"gat_expected_rate", "solo_rate"}

    def __init__(self, channel_model, priority_mode="gat_expected_rate"):
        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(
                f"Unknown priority_mode={priority_mode}; "
                f"expected one of {sorted(self.PRIORITY_MODES)}"
            )
        self.channel_model = channel_model
        self.priority_mode = priority_mode
        self._channel_rate_cache = {}

    def eval(self):
        if hasattr(self.channel_model, "eval"):
            self.channel_model.eval()
        return self

    def clear_cache(self):
        self._channel_rate_cache.clear()

    def _expected_rate(self, state, device):
        key = (id(state), str(device), self.priority_mode)
        cached = self._channel_rate_cache.get(key)
        if cached is not None and cached[0] is state:
            return cached[1]

        direct, source, target, edge_gain = local_edge_gains(state)
        noise = torch.as_tensor(state.noise_power, dtype=torch.float32).cpu()
        snr_gap = float(getattr(state, "snr_gap", 1.0))

        if self.priority_mode == "solo_rate":
            rate = torch.log2(1.0 + direct / noise / snr_gap).to(device)
            self._channel_rate_cache[key] = (state, rate)
            return rate

        with torch.no_grad():
            if hasattr(self.channel_model, "schedule_logits_with_state"):
                logits, _ = self.channel_model.schedule_logits_with_state(state)
                logits = logits.to(device)
            else:
                logits, _ = self.channel_model.schedule_logits(
                    state.vertex_features[:, :1].to(device),
                    state.edge_index.to(device),
                    state.edge_weights.to(device),
                )
        logits = logits.detach()

        expected_interference = torch.zeros_like(direct)
        if source.numel() > 0:
            expected_activity = torch.sigmoid(logits.reshape(-1)).cpu()
            expected_interference.index_add_(
                0, target, edge_gain * expected_activity[source]
            )
        rate = torch.log2(
            1.0
            + direct / (expected_interference + noise) / snr_gap
        ).to(device)
        self._channel_rate_cache[key] = (state, rate)
        return rate

    def schedule_with_state(self, state, queues):
        queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1)
        if self.channel_model is not None:
            try:
                device = next(self.channel_model.parameters()).device
            except (AttributeError, StopIteration):
                device = queues.device
        else:
            device = queues.device
        queues = queues.to(device)
        edge_index = state.edge_index.to(device)
        priority = queues * self._expected_rate(state, device)

        winners = select_local_maxima(
            priority, queues > 0, edge_index
        )
        return winners.to(torch.float32).unsqueeze(1)


class FixedChannelRateLocalMaxScheduler:
    """Diagnostic LocalMax using a precomputed channel-only rate prior."""

    def __init__(self, rate_prior):
        self.rate_prior = torch.as_tensor(
            rate_prior, dtype=torch.float32
        ).reshape(-1)

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1)
        rate = self.rate_prior.to(queues.device)
        if rate.numel() != queues.numel():
            raise ValueError(
                f"rate_prior has {rate.numel()} links, queues has {queues.numel()}"
            )
        priority = queues * self.rate_prior.to(queues.device)
        winners = select_local_maxima(
            priority, queues > 0, state.edge_index.to(queues.device)
        )
        return winners.float().unsqueeze(1)


class ChannelGATHardActionSkeleton:
    """Use the conference GAT output with its original binary-action meaning.

    The conference model was trained so that ``sigmoid(logit)`` is thresholded
    into a channel-only schedule.  This skeleton preserves that semantics and
    applies only the necessary nonempty-queue feasibility mask; it does not
    reinterpret the output as independent neighbor activity probabilities.
    """

    def __init__(self, channel_model):
        self.channel_model = channel_model
        self._action_cache = {}

    def eval(self):
        self.channel_model.eval()
        return self

    def clear_cache(self):
        self._action_cache.clear()

    def schedule_with_state(self, state, queues):
        device = next(self.channel_model.parameters()).device
        key = (id(state), str(device))
        cached = self._action_cache.get(key)
        action = cached[1] if cached is not None and cached[0] is state else None
        if action is None:
            with torch.no_grad():
                logits, _ = self.channel_model.schedule_logits(
                    state.vertex_features[:, :1].to(device),
                    state.edge_index.to(device),
                    state.edge_weights.to(device),
                )
            threshold = float(
                getattr(self.channel_model, "decision_threshold", 0.5)
            )
            probability = torch.sigmoid(logits.reshape(-1))
            action = probability >= threshold
            self._action_cache[key] = (state, action)
        nonempty = torch.as_tensor(queues, device=device).reshape(-1) > 0
        return (action & nonempty).to(torch.float32).unsqueeze(1)


class LocalPhysicalReuseScheduler:
    """LocalMax skeleton followed by one local physical reuse round."""

    def __init__(self, localmax):
        self.localmax = localmax
        self.last_diagnostics = {}

    def eval(self):
        self.localmax.eval()
        return self

    def schedule_with_state(self, state, queues):
        base = self.localmax.schedule_with_state(
            state, queues
        ).reshape(-1).cpu()
        action, margins, joiners = complete_physical_reuse_from_base(
            state, queues, base
        )
        self.last_diagnostics = {
            "positive_candidates": int((margins > 0).sum()),
            "selected_joiners": int(joiners.sum()),
        }
        return action


class FixedChannelRatePhysicalReuseScheduler(LocalPhysicalReuseScheduler):
    """Diagnostic full scheduler for a fixed channel-only rate prior."""

    def __init__(self, rate_prior):
        super().__init__(FixedChannelRateLocalMaxScheduler(rate_prior))


class GATExpectedRateLocalPhysicalReuseScheduler(LocalPhysicalReuseScheduler):
    """GAT expected-rate LocalMax/reuse initializer retained for ablation."""

    model_version = "local_physical_reuse_v2_gat_expected_rate"

    def __init__(self, channel_model):
        if hasattr(channel_model, "reset_temporal_state"):
            channel_model.reset_temporal_state()
        super().__init__(QueueChannelLocalMaxScheduler(
            channel_model, priority_mode="gat_expected_rate"
        ))


class GATHardSeedPhysicalReuseScheduler(LocalPhysicalReuseScheduler):
    """Minimal interface audit: conference hard action plus physical reuse."""

    model_version = "local_physical_reuse_v2_gat_hard_seed_audit"

    def __init__(self, channel_model):
        super().__init__(ChannelGATHardActionSkeleton(channel_model))


class PurePhysicalReuseScheduler(LocalPhysicalReuseScheduler):
    """No-learning ablation using queue times interference-free solo rate."""

    model_version = "local_physical_reuse_ablation_solo_rate"

    def __init__(self, channel_model=None):
        super().__init__(QueueChannelLocalMaxScheduler(
            channel_model, priority_mode="solo_rate"
        ))
