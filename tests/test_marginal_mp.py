import unittest

import torch

from baseline.queue_schedulers import HardConflictLocalGreedyScheduler
from models.marginal_mp import DirectLinkMarginalMP, exact_link_toggle_marginals
from utils.project_setup import build_network_generator, load_config
from utils.scheduling_objective import mws_objective


def _state(seed=123):
    config = load_config("configs/paper.yaml")
    config.update({"NUM_UAV": 8})
    torch.manual_seed(seed)
    return build_network_generator(config).generate_network(
        csi_mode="perfect", include_queue_feature=False
    )


class MarginalMPTest(unittest.TestCase):
    def test_vectorized_toggle_margins_match_brute_force(self):
        state = _state()
        queues = torch.arange(
            1, state.num_vertices + 1, dtype=torch.float32
        ) % 7 + 1
        action = torch.zeros(state.num_vertices)
        action[::5] = 1.0
        margins = exact_link_toggle_marginals(state, queues, action)
        base = mws_objective(action, queues, state)
        for link in range(state.num_vertices):
            toggled = action.clone()
            toggled[link] = 1.0 - toggled[link]
            expected = mws_objective(toggled, queues, state) - base
            self.assertTrue(torch.isclose(
                margins[link], torch.tensor(expected), rtol=2e-4, atol=2e-4
            ))

    def test_scheduler_returns_binary_action_and_overhead_record(self):
        state = _state(seed=456)
        queues = torch.ones(state.num_vertices)
        scheduler = DirectLinkMarginalMP(rounds=4)
        action = scheduler.schedule_with_state(state, queues)
        self.assertEqual(action.shape, (state.num_vertices, 1))
        self.assertTrue(torch.all((action == 0) | (action == 1)))
        diagnostics = scheduler.last_diagnostics
        executed = diagnostics["executed_rounds"]
        self.assertEqual(
            diagnostics["logical_neighbor_exchange_rounds"], 1 + 2 * executed
        )
        self.assertEqual(
            len(diagnostics["physical_externality_pairs_per_round"]), executed
        )

    def test_hard_conflict_lgs_returns_independent_set(self):
        state = _state(seed=789)
        queues = torch.ones(state.num_vertices)
        action = HardConflictLocalGreedyScheduler().schedule_with_state(
            state, queues
        ).reshape(-1).bool()
        source, target = state.edge_index
        self.assertFalse(torch.any(action[source] & action[target]))


if __name__ == "__main__":
    unittest.main()
