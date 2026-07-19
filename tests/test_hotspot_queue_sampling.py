import unittest

from utils.fixed_csi_benchmark import generate_hotspot_arrivals


class HotspotQueueSamplingTest(unittest.TestCase):
    def test_hotspot_trace_has_requested_total_mean(self):
        arrivals, profile = generate_hotspot_arrivals(
            20000, 80, 8.0, arrival_seed=5, hotspot_seed=7
        )
        self.assertTrue((arrivals > 0).any())
        self.assertAlmostEqual(arrivals.sum(1).float().mean().item(), 8.0, delta=0.1)
        self.assertEqual(profile["num_hotspot_links"], 16)


if __name__ == "__main__":
    unittest.main()
