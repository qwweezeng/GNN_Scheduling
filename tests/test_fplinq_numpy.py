import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from baseline.fplinq_numpy import binarize_fplinq, optimize_fplinq
from baseline.queue_schedulers import WeightedFPLinQScheduler
from tests.test_local_physical_reuse import two_link_state


class _LegacyParameters:
    tx_power = 1.0
    output_noise_power = 0.2


class FPLinQNumPyTest(unittest.TestCase):
    def test_matches_retained_update_without_double_power_scaling(self):
        utility_path = Path(
            "baseline/Spatial_Deep_Learning_for_Wireless_Scheduling-master/Utilities"
        ).resolve()
        sys.path.insert(0, str(utility_path))
        try:
            import FPLinQ as retained
        finally:
            sys.path.pop(0)

        received = np.array(
            [[2.0, 0.20, 0.05], [0.10, 1.7, 0.25], [0.08, 0.12, 1.4]],
            dtype=np.float64,
        )
        weights = np.array([1.0, 2.0, 0.5], dtype=np.float64)
        expected = retained.FP_optimize(
            _LegacyParameters(), received[None, :, :], weights[None, :]
        )[0]
        actual = optimize_fplinq(received, 0.2, weights, iterations=100)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_zero_weight_stays_finite_and_inactive(self):
        received = np.array([[1.0, 0.2], [0.1, 1.2]])
        allocation = optimize_fplinq(received, 0.1, [1.0, 0.0])
        self.assertTrue(np.isfinite(allocation).all())
        self.assertEqual(allocation[1], 0.0)
        self.assertEqual(binarize_fplinq(allocation)[1], 0.0)

    def test_queue_scheduler_is_binary_and_records_iterations(self):
        scheduler = WeightedFPLinQScheduler(iterations=8)
        action = scheduler.schedule_with_state(
            two_link_state(), torch.tensor([3.0, 1.0])
        )
        self.assertEqual(action.shape, (2, 1))
        self.assertTrue(torch.all((action == 0.0) | (action == 1.0)))
        self.assertEqual(scheduler.last_diagnostics["iterations"], 8)


if __name__ == "__main__":
    unittest.main()
