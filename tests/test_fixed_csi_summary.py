import unittest

from experiments.summarize_fixed_csi_qos import add_reference_differences


class FixedCSISummaryTest(unittest.TestCase):
    def test_reference_difference_uses_matching_method_and_load(self):
        key = ("GAT-Assisted MarginalMP", 2.4)
        current = {key: {
            "successful_packet_total_rate_pps": 2.0,
            "average_total_queue_backlog_mbit": 3.0,
        }}
        reference = {key: {
            "successful_packet_total_rate_pps": 1.5,
            "average_total_queue_backlog_mbit": 4.0,
        }}
        add_reference_differences(current, reference)
        self.assertEqual(
            current[key]["delta_successful_packet_total_rate_pps"], 0.5
        )
        self.assertEqual(
            current[key]["delta_average_total_queue_backlog_mbit"], -1.0
        )


if __name__ == "__main__":
    unittest.main()
