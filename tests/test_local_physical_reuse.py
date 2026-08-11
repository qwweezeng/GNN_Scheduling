import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.local_physical_reuse import (
    ChannelGATHardActionSkeleton,
    GATExpectedRateLocalPhysicalReuseScheduler,
    LocalPhysicalReuseScheduler,
    QueueChannelLocalMaxScheduler,
    conflict_graph_local_join_margins,
    select_local_positive_maxima,
)
from utils.scheduling_objective import mws_objective


class FixedSkeleton:
    def __init__(self, action):
        self.action = torch.as_tensor(
            action, dtype=torch.float32
        ).reshape(-1, 1)

    def eval(self):
        return self

    def schedule_with_state(self, state, queues):
        del state, queues
        return self.action.clone()


def two_link_state(edge_index=None):
    power = torch.zeros(4, 4)
    power[0, 1] = 8.0
    power[2, 3] = 6.0
    power[0, 3] = 0.6
    power[2, 1] = 0.8
    if edge_index is None:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return SimpleNamespace(
        num_vertices=2,
        signal_power=power,
        signal_power_large_scale=None,
        signal_power_quant=None,
        csi_mode="perfect",
        links=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        noise_power=1.0,
        snr_gap=1.0,
        edge_index=edge_index,
        edge_weights=torch.ones(edge_index.size(1), 1),
    )


class FixedChannelGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.forward_calls = 0

    def schedule_logits(self, x, edge_index, edge_attr):
        del edge_index, edge_attr
        self.forward_calls += 1
        logits = torch.linspace(-0.5, 0.5, x.size(0)).unsqueeze(1)
        return logits, torch.zeros(x.size(0), 2)


class LocalPhysicalReuseTest(unittest.TestCase):
    def test_local_margin_matches_full_two_link_objective_change(self):
        state = two_link_state()
        queues = torch.tensor([3.0, 2.0])
        base = torch.tensor([1.0, 0.0])
        margin = conflict_graph_local_join_margins(
            state, queues, base
        )[1]
        brute_force = (
            mws_objective(torch.tensor([1.0, 1.0]), queues, state)
            - mws_objective(base, queues, state)
        )
        self.assertAlmostEqual(float(margin), brute_force, places=5)

    def test_local_competition_keeps_only_higher_neighbor(self):
        scores = torch.tensor([2.0, 1.0, 0.5])
        valid = torch.ones(3, dtype=torch.bool)
        edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        selected = select_local_positive_maxima(scores, valid, edges)
        self.assertTrue(torch.equal(
            selected, torch.tensor([True, False, True])
        ))

    def test_scheduler_never_activates_empty_queue(self):
        state = two_link_state(torch.empty(2, 0, dtype=torch.long))
        scheduler = LocalPhysicalReuseScheduler(FixedSkeleton([1.0, 0.0]))
        action = scheduler.schedule_with_state(
            state, torch.tensor([1.0, 0.0])
        ).reshape(-1)
        self.assertTrue(torch.equal(action, torch.tensor([1.0, 0.0])))

    def test_frozen_v2_is_queue_scale_invariant(self):
        state = two_link_state()
        state.vertex_features = torch.tensor([[9.0], [7.0]])
        scheduler = GATExpectedRateLocalPhysicalReuseScheduler(
            FixedChannelGAT()
        )
        queues = torch.tensor([3.0, 2.0])
        original = scheduler.schedule_with_state(state, queues)
        scaled = scheduler.schedule_with_state(state, 11.0 * queues)
        self.assertTrue(torch.equal(original, scaled))

    def test_hard_seed_preserves_conference_threshold_semantics(self):
        state = two_link_state()
        state.vertex_features = torch.tensor([[9.0], [7.0]])
        skeleton = ChannelGATHardActionSkeleton(FixedChannelGAT())
        action = skeleton.schedule_with_state(
            state, torch.tensor([1.0, 1.0])
        ).reshape(-1)
        self.assertTrue(torch.equal(action, torch.tensor([0.0, 1.0])))

    def test_gat_rate_prior_is_cached_per_block_but_queues_update_per_slot(self):
        state = two_link_state()
        state.vertex_features = torch.tensor([[9.0], [7.0]])
        model = FixedChannelGAT()
        scheduler = QueueChannelLocalMaxScheduler(model)

        first = scheduler.schedule_with_state(
            state, torch.tensor([100.0, 1.0])
        ).reshape(-1)
        second = scheduler.schedule_with_state(
            state, torch.tensor([1.0, 100.0])
        ).reshape(-1)

        self.assertEqual(model.forward_calls, 1)
        self.assertTrue(torch.equal(first, torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(second, torch.tensor([0.0, 1.0])))

        next_block = two_link_state()
        next_block.vertex_features = torch.tensor([[9.0], [7.0]])
        scheduler.clear_cache()
        scheduler.schedule_with_state(next_block, torch.ones(2))
        self.assertEqual(model.forward_calls, 2)


if __name__ == "__main__":
    unittest.main()
