"""Chapter V: GAT-assisted marginal message passing.

The code follows the order of Algorithm 2 in the paper:

1. build a queue-aware LocalMax schedule;
2. compute one graph-local toggle marginal per link;
3. confirm proposals over closed coordination neighborhoods;
4. confirm joining proposals at both UAV endpoints;
5. apply the confirmed toggles in parallel.

Only direct-link powers and powers carried by conflict-graph edges are used in
the marginal stage.  The full received-power matrix is never used to account
for non-conflict physical externalities.
"""

import torch

from models.initial_schedule_construction import (
    DirectLinkLocalMaxScheduler,
    GATAssistedLocalMaxScheduler,
)
from models.local_physical_reuse import local_edge_gains
from utils.feasibility import shared_uav_action_audit


PAPER_METHOD_NAME = "GAT-Assisted Marginal Message Passing"
PAPER_METHOD_ACRONYM = "GAT-MP"
DIRECT_LINK_METHOD_NAME = "Direct-MP"
GAT_ASSISTED_METHOD_NAME = "GAT-MP"

MARGINAL_PROTOCOL_VERSION = "conflict_graph_local_coordination_v1"


def build_conflict_neighbors(state):
    """Return directed incoming and outgoing conflict neighbors."""
    incoming = [set() for _ in range(int(state.num_vertices))]
    outgoing = [set() for _ in range(int(state.num_vertices))]
    source, target = state.edge_index.detach().cpu()
    for sender, receiver in zip(source.tolist(), target.tolist()):
        outgoing[sender].add(receiver)
        incoming[receiver].add(sender)
    return (
        [tuple(sorted(neighbors)) for neighbors in incoming],
        [tuple(sorted(neighbors)) for neighbors in outgoing],
    )


def build_coordination_neighborhoods(state):
    """Return each link's closed incoming/outgoing conflict neighborhood."""
    incoming, outgoing = build_conflict_neighbors(state)
    return [
        tuple(sorted({link, *incoming[link], *outgoing[link]}))
        for link in range(int(state.num_vertices))
    ]


def build_coordination_memberships(coordination_neighborhoods):
    """Flatten candidate-to-neighborhood membership for vectorized voting."""
    sizes = torch.tensor([
        len(neighborhood) for neighborhood in coordination_neighborhoods
    ], dtype=torch.long)
    candidates = torch.repeat_interleave(
        torch.arange(len(coordination_neighborhoods)), sizes
    )
    members = torch.tensor([
        member
        for neighborhood in coordination_neighborhoods
        for member in neighborhood
    ], dtype=torch.long)
    return candidates, members, sizes


def compute_graph_local_rates(state, action):
    """Compute rates using interference only on conflict-graph edges."""
    active = torch.as_tensor(action, dtype=torch.float32).reshape(-1).cpu()
    direct, source, target, edge_gain = local_edge_gains(state)

    interference = torch.zeros_like(direct)
    if source.numel() > 0:
        interference.index_add_(0, target, edge_gain * active[source])

    noise = float(state.noise_power)
    gap = float(getattr(state, "snr_gap", 1.0))
    return torch.log2(1.0 + direct / (noise + interference) / gap)


def conflict_graph_utility(state, queues, action):
    """Return U_G(A)=sum_i Q_i R_i^G(A) over active links."""
    queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1).cpu()
    active = torch.as_tensor(action, dtype=torch.float32).reshape(-1).cpu()
    rates = compute_graph_local_rates(state, active)
    return torch.sum(queues * active * rates)


def conflict_graph_toggle_marginals(state, queues, action):
    """Return the graph-local utility change from toggling every link.

    The own-service term is combined with rate variations reported by active
    outgoing conflict neighbors.  No non-conflict receiver contributes to the
    result.
    """
    queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1).cpu()
    active = torch.as_tensor(action, dtype=torch.bool).reshape(-1).cpu()
    active_float = active.float()
    direct, source, target, edge_gain = local_edge_gains(state)

    interference = torch.zeros_like(direct)
    if source.numel() > 0:
        interference.index_add_(0, target, edge_gain * active_float[source])

    noise = float(state.noise_power)
    gap = float(getattr(state, "snr_gap", 1.0))
    current_rates = torch.log2(1.0 + direct / (noise + interference) / gap)

    own_service = torch.where(
        active,
        -queues * current_rates,
        queues * current_rates,
    )
    neighbor_variation = torch.zeros_like(own_service)

    if source.numel() > 0:
        interference_after_toggle = torch.where(
            active[source],
            interference[target] - edge_gain,
            interference[target] + edge_gain,
        ).clamp_min(0.0)
        rates_after_toggle = torch.log2(
            1.0 + direct[target] / (noise + interference_after_toggle) / gap
        )
        edge_variation = (
            active_float[target]
            * queues[target]
            * (rates_after_toggle - current_rates[target])
        )
        neighbor_variation.index_add_(0, source, edge_variation)

    return own_service + neighbor_variation


def confirm_disjoint_neighborhood_proposals(
    marginals,
    proposals,
    coordination_neighborhoods,
    coordination_memberships=None,
):
    """Confirm proposals chosen by every member of their neighborhood."""
    marginals = torch.as_tensor(marginals, dtype=torch.float32).reshape(-1).cpu()
    proposals = torch.as_tensor(proposals, dtype=torch.bool).reshape(-1).cpu()
    if coordination_memberships is None:
        coordination_memberships = build_coordination_memberships(
            coordination_neighborhoods
        )
    candidates, members, neighborhood_sizes = coordination_memberships

    active_memberships = proposals[candidates]
    proposal_candidates = candidates[active_memberships]
    proposal_members = members[active_memberships]
    if proposal_candidates.numel() == 0:
        return torch.zeros_like(proposals)

    proposal_marginals = marginals[proposal_candidates]
    best_margin = torch.full_like(marginals, -torch.inf)
    best_margin.scatter_reduce_(
        0, proposal_members, proposal_marginals,
        reduce="amax", include_self=True,
    )

    has_best_margin = proposal_marginals == best_margin[proposal_members]
    tied_members = proposal_members[has_best_margin]
    tied_candidates = proposal_candidates[has_best_margin]

    no_vote = len(coordination_neighborhoods)
    member_votes = torch.full(
        (no_vote,), no_vote, dtype=torch.long
    )
    member_votes.scatter_reduce_(
        0, tied_members, tied_candidates,
        reduce="amin", include_self=True,
    )

    vote_count = torch.bincount(
        member_votes[member_votes < no_vote], minlength=no_vote
    )
    return proposals & (vote_count == neighborhood_sizes)


def confirm_endpoint_join_proposals(state, marginals, proposals, action):
    """Require both free endpoint UAVs to confirm a joining proposal."""
    marginals = torch.as_tensor(marginals, dtype=torch.float32).reshape(-1).cpu()
    proposals = torch.as_tensor(proposals, dtype=torch.bool).reshape(-1).cpu()
    active = torch.as_tensor(action, dtype=torch.bool).reshape(-1).cpu()
    links = state.links.detach().cpu()

    occupied = set(links[active].reshape(-1).tolist())
    joining = proposals & ~active
    joining_indices = torch.nonzero(joining, as_tuple=True)[0].tolist()
    joining_indices = [
        link for link in joining_indices
        if int(links[link, 0]) not in occupied
        and int(links[link, 1]) not in occupied
    ]

    endpoint_vote = {}
    for candidate in joining_indices:
        for endpoint in links[candidate].tolist():
            current = endpoint_vote.get(endpoint)
            if current is None or (
                float(marginals[candidate]), -candidate
            ) > (
                float(marginals[current]), -current
            ):
                endpoint_vote[endpoint] = candidate

    confirmed = active & proposals
    for candidate in joining_indices:
        tx, rx = links[candidate].tolist()
        confirmed[candidate] = (
            endpoint_vote.get(tx) == candidate
            and endpoint_vote.get(rx) == candidate
        )
    return confirmed


def neighborhoods_are_disjoint(selected, coordination_neighborhoods):
    """Return whether selected links have pairwise-disjoint neighborhoods."""
    used = set()
    for candidate in torch.nonzero(selected, as_tuple=True)[0].tolist():
        neighborhood = set(coordination_neighborhoods[candidate])
        if used.intersection(neighborhood):
            return False
        used.update(neighborhood)
    return True


class MarginalMPScheduler:
    """Marginal coordination with a pluggable LocalMax initializer."""

    model_version = MARGINAL_PROTOCOL_VERSION
    paper_name = PAPER_METHOD_NAME
    paper_acronym = PAPER_METHOD_ACRONYM

    def __init__(self, initializer, rounds=4, initializer_name=None):
        self.initializer = initializer
        self.rounds = int(rounds)
        self.initializer_name = str(
            initializer_name
            or getattr(initializer, "model_version", type(initializer).__name__)
        )
        self.last_diagnostics = {}
        self._neighborhood_state = None
        self._coordination_neighborhoods = None
        self._coordination_memberships = None

    def eval(self):
        if hasattr(self.initializer, "eval"):
            self.initializer.eval()
        return self

    def clear_cache(self):
        if hasattr(self.initializer, "clear_cache"):
            self.initializer.clear_cache()
        self._neighborhood_state = None
        self._coordination_neighborhoods = None
        self._coordination_memberships = None

    def _coordination_for(self, state):
        """Build the fixed conflict-graph neighborhoods once per block."""
        if state is not self._neighborhood_state:
            self._coordination_neighborhoods = (
                build_coordination_neighborhoods(state)
            )
            self._coordination_memberships = build_coordination_memberships(
                self._coordination_neighborhoods
            )
            self._neighborhood_state = state
        return (
            self._coordination_neighborhoods,
            self._coordination_memberships,
        )

    def schedule_with_state(self, state, queues):
        action = self.initializer.schedule_with_state(
            state, queues
        ).reshape(-1).round().bool().cpu()
        neighborhoods, neighborhood_memberships = self._coordination_for(state)
        neighborhood_sizes = neighborhood_memberships[2]

        initial_audit = shared_uav_action_audit(action, state.links)
        if not initial_audit["feasible"]:
            raise RuntimeError("MarginalMP initializer must return a feasible action")

        utility_per_round = [
            float(conflict_graph_utility(state, queues, action))
        ]
        positive_per_round = []
        accepted_per_round = []
        accepted_indices_per_round = []
        accepted_joins_per_round = []
        accepted_leaves_per_round = []
        disjoint_per_round = []
        active_per_round = [int(action.sum())]
        rate_variation_messages = []
        proposal_messages = []
        confirmation_messages = []
        endpoint_confirmations = []

        source, target = state.edge_index.detach().cpu()
        for _ in range(self.rounds):
            marginals = conflict_graph_toggle_marginals(
                state, queues, action
            )
            proposals = torch.isfinite(marginals) & (marginals > 1e-6)

            neighborhood_ok = confirm_disjoint_neighborhood_proposals(
                marginals,
                proposals,
                neighborhoods,
                neighborhood_memberships,
            )
            endpoint_ok = confirm_endpoint_join_proposals(
                state, marginals, proposals, action
            )
            confirmed = neighborhood_ok & endpoint_ok

            positive_per_round.append(int(proposals.sum()))
            accepted_per_round.append(int(confirmed.sum()))
            accepted_indices_per_round.append(
                torch.nonzero(confirmed, as_tuple=True)[0].tolist()
            )
            accepted_joins_per_round.append(int((confirmed & ~action).sum()))
            accepted_leaves_per_round.append(int((confirmed & action).sum()))
            disjoint_per_round.append(
                neighborhoods_are_disjoint(confirmed, neighborhoods)
            )
            proposal_messages.append(int(neighborhood_sizes[proposals].sum()))
            confirmation_messages.append(int(
                neighborhood_sizes[neighborhood_ok].sum()
            ))
            endpoint_confirmations.append(int((confirmed & ~action).sum()) * 2)
            rate_variation_messages.append(int(
                (action[target] & proposals[source]).sum()
                if source.numel() > 0 else 0
            ))

            if not confirmed.any():
                break

            action = torch.logical_xor(action, confirmed)
            round_audit = shared_uav_action_audit(action, state.links)
            if not round_audit["feasible"]:
                raise RuntimeError("MarginalMP endpoint confirmation failed")

            utility_per_round.append(
                float(conflict_graph_utility(state, queues, action))
            )
            active_per_round.append(int(action.sum()))

        self.last_diagnostics = {
            "method": self.paper_acronym,
            "protocol_version": MARGINAL_PROTOCOL_VERSION,
            "initializer": self.initializer_name,
            "configured_rounds": self.rounds,
            "executed_rounds": len(accepted_per_round),
            "coordination_neighborhood_sizes": [
                int(size) for size in neighborhood_sizes
            ],
            "positive_proposals_per_round": positive_per_round,
            "accepted_toggles_per_round": accepted_per_round,
            "accepted_link_indices_per_round": accepted_indices_per_round,
            "accepted_joins_per_round": accepted_joins_per_round,
            "accepted_leaves_per_round": accepted_leaves_per_round,
            "accepted_neighborhoods_disjoint_per_round": disjoint_per_round,
            "active_links_per_round": active_per_round,
            "conflict_graph_utility_per_round": utility_per_round,
            "conflict_graph_utility_delta_per_round": [
                after - before
                for before, after in zip(
                    utility_per_round, utility_per_round[1:]
                )
            ],
            "rate_variation_messages_per_round": rate_variation_messages,
            "proposal_messages_per_round": proposal_messages,
            "confirmation_messages_per_round": confirmation_messages,
            "endpoint_confirmations_per_round": endpoint_confirmations,
            "hard_feasibility_initial": initial_audit,
            "hard_feasibility_final": shared_uav_action_audit(
                action, state.links
            ),
        }
        return action.float().unsqueeze(1)


class DirectLinkMarginalMP(MarginalMPScheduler):
    """Direct-link LocalMax followed by graph-local marginal coordination."""

    model_version = "direct_mp_local_coordination_v1"
    paper_variant_name = DIRECT_LINK_METHOD_NAME

    def __init__(self, rounds=4):
        super().__init__(
            DirectLinkLocalMaxScheduler(),
            rounds=rounds,
            initializer_name="Direct-Link LocalMax",
        )


class GATAssistedMarginalMP(MarginalMPScheduler):
    """GAT prior plus graph-local marginal coordination."""

    model_version = "gat_mp_local_coordination_v1"
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
    "MARGINAL_PROTOCOL_VERSION",
    "build_conflict_neighbors",
    "build_coordination_neighborhoods",
    "build_coordination_memberships",
    "compute_graph_local_rates",
    "conflict_graph_utility",
    "conflict_graph_toggle_marginals",
    "confirm_disjoint_neighborhood_proposals",
    "confirm_endpoint_join_proposals",
    "neighborhoods_are_disjoint",
    "MarginalMPScheduler",
    "DirectLinkMarginalMP",
    "GATAssistedMarginalMP",
]
