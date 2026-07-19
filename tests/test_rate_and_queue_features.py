import unittest
from types import SimpleNamespace

import torch

from models.model import calculate_achievable_rate
from utils.queue_features import local_relative_queue


class RateFormulaTest(unittest.TestCase):
    def _state(self):
        return SimpleNamespace(
            num_vertices=1,
            signal_power=torch.tensor([[4.0]]),
            signal_power_large_scale=None,
            signal_power_quant=None,
            csi_mode="perfect",
            links=torch.tensor([[0, 0]], dtype=torch.long),
            noise_power=1.0,
            snr_gap=1.0,
        )

    def test_continuous_pi_multiplies_own_rate_once(self):
        state = self._state()
        full = calculate_achievable_rate(
            torch.tensor([[1.0]]), state, binary=False
        )
        half = calculate_achievable_rate(
            torch.tensor([[0.5]]), state, binary=False
        )
        self.assertTrue(torch.allclose(half, 0.5 * full, atol=1e-7))

    def test_zero_pi_has_zero_rate(self):
        rate = calculate_achievable_rate(
            torch.tensor([[0.0]]), self._state(), binary=False
        )
        self.assertEqual(rate.item(), 0.0)


class LocalQueueFeatureTest(unittest.TestCase):
    def test_uses_only_one_hop_maximum(self):
        queues = torch.tensor([2.0, 4.0, 10.0])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        feature = local_relative_queue(queues, edge_index)
        expected = torch.tensor([0.5, 1.0, 1.0])
        self.assertTrue(torch.allclose(feature, expected))


if __name__ == "__main__":
    unittest.main()
