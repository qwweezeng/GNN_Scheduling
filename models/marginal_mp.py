"""Marginal-Utility Message-Passing Scheduling (MarginalMP).

MarginalMP refines a reliable binary initial schedule for a fixed number of
rounds.  In every round, each inactive link evaluates the exact queue-weighted
utility change caused by joining the current active set, while each active link
evaluates the exact change caused by leaving it.  Positive proposals contend
with conflict neighbors, and conflict-local maxima update in parallel.

The implementation is model-driven rather than a learned MPNN.  The paper-main
variant reuses the channel GAT in its initial-schedule constructor; all
marginal-utility messages below are still derived directly from the observed
SINR model.  The Direct-Link initializer is retained as the mandatory no-GAT
ablation.
"""

import torch

from models.initial_schedule_construction import (
    DirectLinkLocalMaxScheduler,
    GATAssistedLocalMaxScheduler,
)
from models.local_physical_reuse import (
    _signal_power_matrix,
    select_local_positive_maxima,
)


PAPER_METHOD_NAME = "Marginal-Utility Message-Passing Scheduling"
PAPER_METHOD_ACRONYM = "MarginalMP"
DIRECT_LINK_METHOD_NAME = "Direct-Link MarginalMP"
GAT_ASSISTED_METHOD_NAME = "GAT-Assisted MarginalMP"


def exact_link_toggle_marginals(state, queues, action):
    """Return the exact utility change from toggling every individual link.

    For the current active set A, an inactive link receives
    ``U(A union {i}) - U(A)`` and an active link receives
    ``U(A without {i}) - U(A)``.  Only active receivers have nonzero
    externality terms, so the physical-message tensors have O(L|A|), rather
    than O(L^2), storage.
    """
    queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1).cpu()
    active = torch.as_tensor(action, dtype=torch.bool).reshape(-1).cpu()
    signal_power = _signal_power_matrix(state)
    links = state.links.detach().cpu()
    link_tx, link_rx = links[:, 0], links[:, 1]
    direct = signal_power[link_tx, link_rx].float()
    noise = float(state.noise_power)
    gap = float(getattr(state, "snr_gap", 1.0))
    active_indices = torch.nonzero(active, as_tuple=True)[0]

    if active_indices.numel() > 0:
        gain_from_active = signal_power[
            link_tx[active_indices, None], link_rx[None, :]
        ].float()
        gain_from_active[
            torch.arange(active_indices.numel()), active_indices
        ] = 0.0
        interference = gain_from_active.sum(dim=0)

        gain_to_active = signal_power[
            link_tx[:, None], link_rx[active_indices][None, :]
        ].float()
        gain_to_active[
            active_indices, torch.arange(active_indices.numel())
        ] = 0.0
    else:
        interference = torch.zeros_like(direct)
        gain_to_active = torch.empty(
            (queues.numel(), 0), dtype=direct.dtype
        )

    rates = torch.log2(1.0 + direct / (interference + noise) / gap)
    if active_indices.numel() > 0:
        active_direct = direct[active_indices]
        active_interference = interference[active_indices]
        active_rates = rates[active_indices]
        active_queues = queues[active_indices]

        rates_after_add = torch.log2(
            1.0
            + active_direct.unsqueeze(0)
            / (active_interference.unsqueeze(0) + gain_to_active + noise)
            / gap
        )
        rates_after_remove = torch.log2(
            1.0
            + active_direct.unsqueeze(0)
            / (
                (active_interference.unsqueeze(0) - gain_to_active).clamp_min(0.0)
                + noise
            )
            / gap
        )
        add_externality = (
            (rates_after_add - active_rates.unsqueeze(0))
            * active_queues.unsqueeze(0)
        ).sum(dim=1)
        remove_externality = (
            (rates_after_remove - active_rates.unsqueeze(0))
            * active_queues.unsqueeze(0)
        ).sum(dim=1)
    else:
        add_externality = torch.zeros_like(queues)
        remove_externality = torch.zeros_like(queues)

    join_marginals = queues * rates + add_externality
    leave_marginals = -queues * rates + remove_externality
    marginals = torch.where(active, leave_marginals, join_marginals)
    return torch.where(
        queues > 0,
        marginals,
        torch.full_like(marginals, -float("inf")),
    )


class MarginalMPScheduler:
    """Generic MarginalMP with a pluggable binary schedule initializer."""

    model_version = "marginal_mp_v1"
    paper_name = PAPER_METHOD_NAME
    paper_acronym = PAPER_METHOD_ACRONYM

    def __init__(self, initializer, rounds=4, initializer_name=None):
        if int(rounds) < 1:
            raise ValueError("rounds must be positive")
        if not hasattr(initializer, "schedule_with_state"):
            raise TypeError("initializer must implement schedule_with_state")
        self.initializer = initializer
        self.rounds = int(rounds)
        self.initializer_name = str(
            initializer_name
            or getattr(initializer, "model_version", type(initializer).__name__)
        )
        self.last_diagnostics = {}

    def eval(self):
        if hasattr(self.initializer, "eval"):
            self.initializer.eval()
        return self

    def schedule_with_state(self, state, queues):
        action = self.initializer.schedule_with_state(
            state, queues
        ).reshape(-1).round().bool().cpu()
        selected_per_round = []
        positive_per_round = []
        active_per_round = [int(action.sum())]
        physical_externality_pairs = []
        direct_service_reports = []
        arbitration_directed_edge_messages = []
        source, target = state.edge_index.detach().cpu()
        for _ in range(self.rounds):
            active_count = int(action.sum())
            num_links = int(action.numel())
            # Deployment-accounting proxy for the exact marginal formula:
            # every candidate needs one own-rate report and an externality
            # contribution from every other currently active receiver.  CSI
            # acquisition itself is accounted separately by the experiment.
            physical_externality_pairs.append(
                active_count * max(num_links - 1, 0)
            )
            direct_service_reports.append(num_links)
            marginals = exact_link_toggle_marginals(state, queues, action)
            valid = torch.isfinite(marginals) & (marginals > 1e-6)
            arbitration_directed_edge_messages.append(int(
                (valid[source] & valid[target]).sum()
                if source.numel() > 0 else 0
            ))
            selected = select_local_positive_maxima(
                marginals, valid, state.edge_index
            )
            positive_per_round.append(int(valid.sum()))
            selected_per_round.append(int(selected.sum()))
            if not selected.any():
                break
            action = torch.logical_xor(action, selected.cpu())
            active_per_round.append(int(action.sum()))
        self.last_diagnostics = {
            "method": self.paper_acronym,
            "initializer": self.initializer_name,
            "configured_rounds": self.rounds,
            "executed_rounds": len(selected_per_round),
            "positive_proposals_per_round": positive_per_round,
            "selected_toggles_per_round": selected_per_round,
            "active_links_per_round": active_per_round,
            "final_active_links": int(action.sum()),
            "logical_neighbor_exchange_rounds": 1 + 2 * len(selected_per_round),
            "round_accounting_definition": (
                "one LocalMax score exchange plus, per MarginalMP iteration, "
                "one physical-marginal feedback phase and one proposal "
                "arbitration phase; CSI acquisition is excluded"
            ),
            "physical_externality_pairs_per_round": physical_externality_pairs,
            "direct_service_reports_per_round": direct_service_reports,
            "arbitration_directed_edge_messages_per_round": (
                arbitration_directed_edge_messages
            ),
        }
        return action.float().unsqueeze(1)


class DirectLinkMarginalMP(MarginalMPScheduler):
    """Mandatory no-GAT ablation: Direct-Link LocalMax plus MarginalMP."""

    model_version = "direct_link_marginal_mp_v2_no_join"
    paper_variant_name = DIRECT_LINK_METHOD_NAME

    def __init__(self, rounds=4):
        super().__init__(
            DirectLinkLocalMaxScheduler(),
            rounds=rounds,
            initializer_name="Direct-Link LocalMax",
        )


class GATAssistedMarginalMP(MarginalMPScheduler):
    """Paper-main GAT-Assisted LocalMax plus unchanged MarginalMP rounds."""

    model_version = "gat_assisted_marginal_mp_v2_no_join"
    paper_variant_name = GAT_ASSISTED_METHOD_NAME

    def __init__(self, channel_model, rounds=4):
        super().__init__(
            GATAssistedLocalMaxScheduler(channel_model),
            rounds=rounds,
            initializer_name="GAT-Assisted LocalMax",
        )


__all__ = [
    "PAPER_METHOD_NAME",
    "PAPER_METHOD_ACRONYM",
    "DIRECT_LINK_METHOD_NAME",
    "GAT_ASSISTED_METHOD_NAME",
    "exact_link_toggle_marginals",
    "MarginalMPScheduler",
    "DirectLinkMarginalMP",
    "GATAssistedMarginalMP",
]
