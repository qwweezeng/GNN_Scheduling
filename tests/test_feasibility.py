import unittest

import numpy as np
import torch

from utils.feasibility import (
    decode_endpoint_mutual_matching,
    project_shared_uav_feasible,
    shared_uav_action_audit,
)


class SharedUavFeasibilityTests(unittest.TestCase):
    def test_audit_detects_tx_rx_and_rx_rx_reuse(self):
        links = torch.tensor([[0, 1], [1, 2], [3, 1], [4, 5]])
        audit = shared_uav_action_audit([1, 1, 1, 0], links)
        self.assertFalse(audit["feasible"])
        self.assertEqual(audit["violating_uavs"], [1])
        self.assertEqual(audit["max_active_links_per_uav"], 3)

    def test_projection_uses_priority_and_preserves_column_shape(self):
        links = torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5]])
        action = torch.ones(4, 1)
        projected, metadata = project_shared_uav_feasible(
            action, links, priorities=np.array([1.0, 2.0, 3.0, 4.0])
        )
        self.assertEqual(tuple(projected.shape), (4, 1))
        self.assertEqual(projected.reshape(-1).tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertTrue(metadata["projected"]["feasible"])
        self.assertEqual(metadata["rejected_project_link_indices"], [2, 0])

    def test_projection_rejects_nonbinary_action(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            project_shared_uav_feasible([0.2, 1.0], [[0, 1], [2, 3]])

    def test_endpoint_mutual_decoder_is_feasible_and_round_limited(self):
        links = torch.tensor([[0, 1], [1, 2], [2, 3], [4, 5]])
        action = torch.ones(4)
        priority = torch.tensor([4.0, 3.0, 2.0, 1.0])
        one_round, one_meta = decode_endpoint_mutual_matching(
            action, links, priorities=priority, max_rounds=1
        )
        self.assertEqual(one_round.tolist(), [1.0, 0.0, 0.0, 1.0])
        self.assertTrue(one_meta["projected"]["feasible"])
        self.assertEqual(one_meta["rounds_executed"], 1)

        full, full_meta = decode_endpoint_mutual_matching(
            action, links, priorities=priority
        )
        self.assertEqual(full.tolist(), [1.0, 0.0, 1.0, 1.0])
        self.assertTrue(full_meta["projected"]["feasible"])
        self.assertEqual(full_meta["unresolved_candidate_link_indices"], [])

    def test_endpoint_mutual_decoder_uses_stable_tie_break(self):
        links = [[0, 1], [0, 2], [1, 3]]
        decoded, _ = decode_endpoint_mutual_matching(
            [1, 1, 1], links, priorities=[1.0, 1.0, 1.0], max_rounds=1
        )
        self.assertEqual(decoded.tolist(), [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
