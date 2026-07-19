import unittest

import torch

from baseline.queue_schedulers import centralized_mws_greedy
from tests.test_local_physical_reuse import two_link_state
from utils.scheduling_objective import mws_objective


class CentralizedMWSGreedyTest(unittest.TestCase):
    def test_output_is_one_flip_local_optimum(self):
        state = two_link_state()
        queues = torch.tensor([3.0, 2.0])
        action = centralized_mws_greedy(state, queues).reshape(-1)
        utility = mws_objective(action, queues, state)
        for index in range(action.numel()):
            candidate = action.clone()
            candidate[index] = 1.0 - candidate[index]
            self.assertLessEqual(
                mws_objective(candidate, queues, state),
                utility + 1e-5,
            )


if __name__ == "__main__":
    unittest.main()
