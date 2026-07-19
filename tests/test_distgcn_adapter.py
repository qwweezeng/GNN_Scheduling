import unittest

import torch

from baseline.distgcn_adapter import build_hardened_mwis_instance
from tests.test_local_physical_reuse import two_link_state


class DistGCNAdapterTest(unittest.TestCase):
    def test_hard_graph_is_symmetric_and_has_queue_weights(self):
        state = two_link_state()
        adjacency, weights, metadata = build_hardened_mwis_instance(
            state, torch.tensor([3.0, 0.0]), inr_threshold_db=-10.0
        )
        self.assertTrue(torch.equal(adjacency, adjacency.T))
        self.assertFalse(torch.diagonal(adjacency).any())
        self.assertGreater(weights[0].item(), 0.0)
        self.assertEqual(weights[1].item(), 0.0)
        self.assertEqual(metadata["num_nodes"], 2)


if __name__ == "__main__":
    unittest.main()
