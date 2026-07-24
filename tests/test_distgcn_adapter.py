import unittest

import torch

from baseline.distgcn_adapter import (
    LinkIndexRoundTrip,
    build_hardened_mwis_instance,
    distgcn_local_greedy,
    to_distgcn_numpy,
)
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

    def test_link_index_roundtrip_is_exact(self):
        mapping = LinkIndexRoundTrip.from_project_indices(6, [4, 1, 5])
        values = torch.arange(6).numpy()
        self.assertEqual(mapping.project_to_third_party(values).tolist(), [4, 1, 5])
        action = mapping.third_party_to_project_action([1, 0, 1])
        self.assertEqual(action.tolist(), [0, 0, 0, 0, 1, 1])

    def test_numpy_conversion_preserves_selected_link_order(self):
        state = two_link_state()
        adjacency, weights, _ = build_hardened_mwis_instance(
            state, torch.tensor([3.0, 2.0]), inr_threshold_db=-10.0
        )
        converted, converted_weights, mapping = to_distgcn_numpy(
            adjacency, weights, project_indices=[1, 0]
        )
        self.assertEqual(converted.shape, (2, 2))
        self.assertEqual(converted_weights.shape, (2, 1))
        self.assertEqual(mapping.third_party_to_project.tolist(), [1, 0])

    def test_project_conflict_graph_is_supported(self):
        state = two_link_state()
        adjacency, _, metadata = build_hardened_mwis_instance(
            state,
            torch.tensor([3.0, 2.0]),
            graph_source="project_conflict",
        )
        self.assertTrue(torch.equal(adjacency, adjacency.T))
        self.assertEqual(metadata["graph_source"], "project_conflict")

    def test_distgcn_postprocessor_is_independent(self):
        adjacency = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]).numpy()
        action, diagnostics = distgcn_local_greedy(
            adjacency, [3.0, 2.0, 1.0]
        )
        self.assertEqual(action.tolist(), [1.0, 0.0, 1.0])
        self.assertGreaterEqual(diagnostics["contention_rounds"], 1)


if __name__ == "__main__":
    unittest.main()
