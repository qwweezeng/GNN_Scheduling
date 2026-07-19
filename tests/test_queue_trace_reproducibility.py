import unittest

import torch

from utils.fixed_csi_benchmark import (
    arrival_trace_seed,
    generate_hotspot_arrivals,
    lambda_key,
)


class QueueTraceReproducibilityTest(unittest.TestCase):
    def test_seed_depends_on_lambda_value_not_list_position(self):
        seed_alone = arrival_trace_seed(42, 2.0)
        seed_in_sweep = arrival_trace_seed(42, [0.5, 1.0, 2.0, 4.0][2])
        self.assertEqual(seed_alone, seed_in_sweep)

    def test_same_mc_seed_and_lambda_reproduce_arrivals(self):
        seed = arrival_trace_seed(43, 2.0)
        first, first_profile = generate_hotspot_arrivals(
            100, 20, 2.0, arrival_seed=seed, hotspot_seed=43
        )
        second, second_profile = generate_hotspot_arrivals(
            100, 20, 2.0, arrival_seed=seed, hotspot_seed=43
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(
            first_profile["hotspot_indices"], second_profile["hotspot_indices"]
        )

    def test_trace_lambda_key_normalizes_equivalent_values(self):
        self.assertEqual(lambda_key(2), lambda_key(2.0))
        self.assertEqual(lambda_key("0.5"), "0.5")


if __name__ == "__main__":
    unittest.main()
