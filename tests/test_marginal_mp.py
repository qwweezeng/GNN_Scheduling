import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from baseline.queue_schedulers import HardConflictLocalGreedyScheduler
from models.marginal_mp import (
    DirectLinkMarginalMP,
    MarginalMPScheduler,
    build_coordination_neighborhoods,
    confirm_disjoint_neighborhood_proposals,
    confirm_endpoint_join_proposals,
    conflict_graph_toggle_marginals,
    conflict_graph_utility,
    neighborhoods_are_disjoint,
)
from utils.feasibility import shared_uav_action_audit
from utils.project_setup import build_network_generator, load_config


def generated_state(seed=123):
    config = load_config("configs/paper.yaml")
    config.update({"NUM_UAV": 8})
    torch.manual_seed(seed)
    return build_network_generator(config).generate_network(
        csi_mode="perfect", include_queue_feature=False
    )


def small_state(links, direct, edges=(), edge_gains=()):
    links = torch.tensor(links, dtype=torch.long)
    num_uav = int(links.max()) + 1
    power = torch.zeros(num_uav, num_uav)
    for link, gain in enumerate(direct):
        tx, rx = links[link]
        power[tx, rx] = float(gain)
    for (source, target), gain in zip(edges, edge_gains):
        tx = links[source, 0]
        rx = links[target, 1]
        power[tx, rx] = float(gain)
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )
    return SimpleNamespace(
        num_vertices=len(links),
        links=links,
        edge_index=edge_index,
        edge_weights=torch.ones((edge_index.size(1), 1)),
        signal_power=power,
        signal_power_large_scale=power.clone(),
        signal_power_quant=power.clone(),
        csi_mode="perfect",
        noise_power=1.0,
        snr_gap=1.0,
        vertex_features=torch.tensor(direct).float().unsqueeze(1),
    )


class FixedInitializer:
    model_version = "fixed_test_initializer"

    def __init__(self, action):
        self.action = torch.as_tensor(
            action, dtype=torch.float32
        ).clone().reshape(-1, 1)

    def schedule_with_state(self, state, queues):
        return self.action.clone()


class MarginalMPTest(unittest.TestCase):
    def test_toggle_marginals_match_graph_local_utility_changes(self):
        state = small_state(
            links=[(0, 1), (2, 3), (4, 5)],
            direct=[8.0, 6.0, 5.0],
            edges=[(0, 1), (2, 1)],
            edge_gains=[0.8, 0.4],
        )
        queues = torch.tensor([3.0, 5.0, 2.0])
        action = torch.tensor([0.0, 1.0, 0.0])
        margins = conflict_graph_toggle_marginals(state, queues, action)
        base = conflict_graph_utility(state, queues, action)

        for link in range(state.num_vertices):
            toggled = action.clone()
            toggled[link] = 1.0 - toggled[link]
            expected = conflict_graph_utility(state, queues, toggled) - base
            self.assertTrue(torch.isclose(
                margins[link], expected, rtol=2e-5, atol=2e-5
            ))

    def test_non_conflict_power_does_not_change_local_marginals(self):
        state = small_state(
            links=[(0, 1), (2, 3), (4, 5)],
            direct=[8.0, 6.0, 5.0],
            edges=[(0, 1)],
            edge_gains=[0.8],
        )
        queues = torch.tensor([3.0, 5.0, 2.0])
        action = torch.tensor([0.0, 1.0, 1.0])
        before = conflict_graph_toggle_marginals(state, queues, action)

        # Link 2 strongly interferes with link 0 in the physical matrix, but
        # (2, 0) is not a conflict edge and must remain invisible to MP.
        state.signal_power[state.links[2, 0], state.links[0, 1]] = 1e6
        after = conflict_graph_toggle_marginals(state, queues, action)
        self.assertTrue(torch.equal(before, after))

    def test_marginal_uses_observed_csi_representation(self):
        def observed_state(true_cross_gain, observed_cross_gain):
            state = small_state(
                links=[(0, 1), (2, 3)],
                direct=[8.0, 6.0],
                edges=[(0, 1)],
                edge_gains=[0.1],
            )
            state.csi_mode = "large_scale"
            state.signal_power[state.links[0, 0], state.links[1, 1]] = (
                true_cross_gain
            )
            state.signal_power_large_scale[
                state.links[0, 0], state.links[1, 1]
            ] = observed_cross_gain
            return state

        queues = torch.tensor([2.0, 5.0])
        action = torch.tensor([0.0, 1.0])
        first = conflict_graph_toggle_marginals(
            observed_state(1.0, 0.2), queues, action
        )
        hidden_true_change = conflict_graph_toggle_marginals(
            observed_state(1e6, 0.2), queues, action
        )
        observed_change = conflict_graph_toggle_marginals(
            observed_state(1.0, 20.0), queues, action
        )

        self.assertTrue(torch.allclose(first, hidden_true_change))
        self.assertFalse(torch.allclose(first, observed_change))

    def test_join_and_leave_are_both_applied_by_xor(self):
        join_state = small_state(
            links=[(0, 1), (2, 3)], direct=[8.0, 6.0]
        )
        join_scheduler = MarginalMPScheduler(
            FixedInitializer([1.0, 0.0]), rounds=1
        )
        joined = join_scheduler.schedule_with_state(
            join_state, torch.tensor([1.0, 1.0])
        ).reshape(-1)
        self.assertTrue(torch.equal(joined, torch.tensor([1.0, 1.0])))

        leave_state = small_state(
            links=[(0, 1), (2, 3)],
            direct=[1.0, 10.0],
            edges=[(0, 1)],
            edge_gains=[100.0],
        )
        leave_scheduler = MarginalMPScheduler(
            FixedInitializer([1.0, 1.0]), rounds=1
        )
        left = leave_scheduler.schedule_with_state(
            leave_state, torch.tensor([0.1, 10.0])
        ).reshape(-1)
        self.assertTrue(torch.equal(left, torch.tensor([0.0, 1.0])))

    def test_overlapping_coordination_neighborhoods_cannot_both_win(self):
        state = small_state(
            links=[(0, 1), (2, 3), (4, 5)],
            direct=[5.0, 5.0, 5.0],
            edges=[(0, 2), (1, 2)],
            edge_gains=[0.2, 0.2],
        )
        neighborhoods = build_coordination_neighborhoods(state)
        proposals = torch.tensor([True, True, False])
        confirmed = confirm_disjoint_neighborhood_proposals(
            torch.tensor([3.0, 2.0, -1.0]), proposals, neighborhoods
        )
        self.assertTrue(torch.equal(
            confirmed, torch.tensor([True, False, False])
        ))
        self.assertTrue(neighborhoods_are_disjoint(confirmed, neighborhoods))

    def test_vectorized_confirmation_matches_direct_voting(self):
        neighborhoods = [
            (0, 1, 3),
            (0, 1, 2),
            (1, 2, 4),
            (0, 3, 4),
            (2, 3, 4),
        ]
        marginals = torch.tensor([2.0, 2.0, 1.0, 3.0, 0.5])

        for bits in range(1 << len(neighborhoods)):
            proposals = torch.tensor([
                bool(bits & (1 << candidate))
                for candidate in range(len(neighborhoods))
            ])
            direct_votes = [-1] * len(neighborhoods)
            for candidate in torch.nonzero(
                proposals, as_tuple=True
            )[0].tolist():
                for member in neighborhoods[candidate]:
                    current = direct_votes[member]
                    if current == -1 or (
                        float(marginals[candidate]), -candidate
                    ) > (
                        float(marginals[current]), -current
                    ):
                        direct_votes[member] = candidate
            expected = torch.tensor([
                bool(proposals[candidate]) and all(
                    direct_votes[member] == candidate
                    for member in neighborhoods[candidate]
                )
                for candidate in range(len(neighborhoods))
            ])

            confirmed = confirm_disjoint_neighborhood_proposals(
                marginals, proposals, neighborhoods
            )
            self.assertTrue(torch.equal(confirmed, expected))

    def test_parallel_marginals_are_additive_for_disjoint_neighborhoods(self):
        state = small_state(
            links=[(0, 1), (2, 3), (4, 5), (6, 7)],
            direct=[10.0, 4.0, 9.0, 5.0],
            edges=[(0, 1), (2, 3)],
            edge_gains=[0.2, 0.3],
        )
        queues = torch.tensor([4.0, 1.0, 3.0, 1.0])
        action = torch.tensor([0.0, 1.0, 0.0, 1.0])
        marginals = conflict_graph_toggle_marginals(state, queues, action)
        neighborhoods = build_coordination_neighborhoods(state)
        selected = confirm_disjoint_neighborhood_proposals(
            marginals, marginals > 0, neighborhoods
        )
        self.assertTrue(neighborhoods_are_disjoint(selected, neighborhoods))

        updated = torch.logical_xor(action.bool(), selected)
        joint_change = (
            conflict_graph_utility(state, queues, updated)
            - conflict_graph_utility(state, queues, action)
        )
        self.assertTrue(torch.isclose(
            joint_change, marginals[selected].sum(), rtol=2e-5, atol=2e-5
        ))
        self.assertGreaterEqual(float(joint_change), 0.0)

    def test_endpoint_confirmation_blocks_occupied_and_competing_joins(self):
        occupied_state = small_state(
            links=[(0, 1), (0, 2)], direct=[5.0, 8.0]
        )
        confirmed = confirm_endpoint_join_proposals(
            occupied_state,
            torch.tensor([-1.0, 10.0]),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
        )
        self.assertFalse(bool(confirmed[1]))

        competing_state = small_state(
            links=[(0, 1), (2, 3), (2, 4)], direct=[5.0, 8.0, 7.0]
        )
        confirmed = confirm_endpoint_join_proposals(
            competing_state,
            torch.tensor([-1.0, 10.0, 9.0]),
            torch.tensor([False, True, True]),
            torch.tensor([True, False, False]),
        )
        self.assertTrue(bool(confirmed[1]))
        self.assertFalse(bool(confirmed[2]))

    def test_scheduler_returns_feasible_action_and_local_diagnostics(self):
        state = generated_state(seed=456)
        queues = torch.ones(state.num_vertices)
        scheduler = DirectLinkMarginalMP(rounds=4)
        action = scheduler.schedule_with_state(state, queues)
        diagnostics = scheduler.last_diagnostics

        self.assertEqual(action.shape, (state.num_vertices, 1))
        self.assertTrue(shared_uav_action_audit(action, state.links)["feasible"])
        utilities = diagnostics["conflict_graph_utility_per_round"]
        self.assertTrue(all(
            after + 1e-5 >= before
            for before, after in zip(utilities, utilities[1:])
        ))
        self.assertIn("rate_variation_messages_per_round", diagnostics)
        self.assertNotIn("physical_externality_pairs_per_round", diagnostics)

    def test_coordination_neighborhoods_are_built_once_per_state(self):
        state = small_state(
            links=[(0, 1), (2, 3)],
            direct=[8.0, 6.0],
            edges=[(0, 1)],
            edge_gains=[0.8],
        )
        scheduler = MarginalMPScheduler(
            FixedInitializer([1.0, 0.0]), rounds=1
        )
        queues = torch.ones(state.num_vertices)

        with patch(
            "models.marginal_mp.build_coordination_neighborhoods",
            wraps=build_coordination_neighborhoods,
        ) as build:
            scheduler.schedule_with_state(state, queues)
            scheduler.schedule_with_state(state, queues * 2)
            self.assertEqual(build.call_count, 1)

            new_state = small_state(
                links=[(0, 1), (2, 3)],
                direct=[8.0, 6.0],
                edges=[(0, 1)],
                edge_gains=[0.8],
            )
            scheduler.schedule_with_state(new_state, queues)
            self.assertEqual(build.call_count, 2)

            scheduler.clear_cache()
            scheduler.schedule_with_state(new_state, queues)
            self.assertEqual(build.call_count, 3)

    def test_infeasible_initializer_is_not_repaired_by_projection(self):
        state = generated_state(seed=654)
        scheduler = MarginalMPScheduler(
            FixedInitializer(torch.ones(state.num_vertices)), rounds=1
        )
        with self.assertRaisesRegex(RuntimeError, "initializer"):
            scheduler.schedule_with_state(
                state, torch.ones(state.num_vertices)
            )

    def test_hard_conflict_lgs_returns_independent_set(self):
        state = generated_state(seed=789)
        queues = torch.ones(state.num_vertices)
        action = HardConflictLocalGreedyScheduler().schedule_with_state(
            state, queues
        ).reshape(-1).bool()
        source, target = state.edge_index
        self.assertFalse(torch.any(action[source] & action[target]))


if __name__ == "__main__":
    unittest.main()
