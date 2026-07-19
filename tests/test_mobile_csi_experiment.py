import unittest

from experiments.evaluate_mobile_block_fading_csi import aggregate, relative_to_latest


def _row(method, delivered, backlog, seed=1):
    return {
        "speed_mps": 5.0,
        "slots_per_block": 5,
        "stale_blocks": 1,
        "arrival_rate": 2.2,
        "seed": seed,
        "method": method,
        "delivered_throughput_mbps": delivered,
        "physical_service_mbps": delivered,
        "successful_delivery_ratio": 1.0,
        "unfinished_packet_fraction": 0.0,
        "average_total_queue_backlog_mbit": backlog,
        "average_packet_delay_ms": 2.0,
        "p95_packet_delay_ms": 3.0,
        "average_active_links": 2.0,
        "mean_csi_absolute_error_db": 0.0,
    }


class MobileCsiExperimentTest(unittest.TestCase):
    def test_aggregate_preserves_method_condition(self):
        summary = aggregate([
            _row("latest", 10.0, 2.0, seed=1),
            _row("latest", 12.0, 4.0, seed=2),
        ])
        key = "v=5|block=5|age=1|Lambda=2.2|method=latest"
        self.assertAlmostEqual(
            summary[key]["delivered_throughput_mbps"]["mean"], 11.0
        )

    def test_relative_loss_uses_paired_latest_trace(self):
        result = relative_to_latest([
            _row("latest", 10.0, 2.0),
            _row("stale", 8.0, 5.0),
        ])
        key = "v=5|block=5|age=1|Lambda=2.2|method=stale"
        self.assertAlmostEqual(
            result[key]["delivered_loss_fraction"]["mean"], 0.2
        )
        self.assertAlmostEqual(
            result[key]["backlog_increase_mbit"]["mean"], 3.0
        )


if __name__ == "__main__":
    unittest.main()
