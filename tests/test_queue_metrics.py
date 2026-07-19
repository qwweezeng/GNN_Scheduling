import unittest
from collections import deque

import torch

from simulator.queue_simulator import QueueSimulator


class PacketDelayTrackingTest(unittest.TestCase):
    def _tracker(self):
        tracker = QueueSimulator.__new__(QueueSimulator)
        tracker.track_packet_delay = True
        tracker._packet_cohorts = [deque()]
        tracker._delivered_packet_delays = []
        tracker._delivered_packet_arrival_slots = []
        tracker.slot = 0
        return tracker

    def test_arrival_is_first_served_in_next_slot(self):
        tracker = self._tracker()
        tracker._update_packet_delay_tracking(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        tracker.slot = 1
        tracker._update_packet_delay_tracking(
            torch.tensor([1.0]), torch.tensor([0.0])
        )
        self.assertEqual(tracker._delivered_packet_delays, [1])

    def test_fractional_service_records_completion_only(self):
        tracker = self._tracker()
        tracker._update_packet_delay_tracking(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        tracker.slot = 1
        tracker._update_packet_delay_tracking(
            torch.tensor([0.4]), torch.tensor([0.0])
        )
        self.assertEqual(tracker._delivered_packet_delays, [])
        tracker.slot = 2
        tracker._update_packet_delay_tracking(
            torch.tensor([0.6]), torch.tensor([0.0])
        )
        self.assertEqual(tracker._delivered_packet_delays, [2])

    def test_deadline_expires_only_after_equal_age_slot(self):
        tracker = self._tracker()
        tracker.device = "cpu"
        tracker.delay_deadline_slots = 2
        tracker.queue_states = torch.tensor([1.0])
        tracker._sync_queue_features = lambda: None
        tracker._update_packet_delay_tracking(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        tracker.slot = 2
        tracker._expire_packet_deadlines()
        self.assertEqual(len(tracker._packet_cohorts[0]), 1)
        tracker.slot = 3
        tracker._expire_packet_deadlines()
        self.assertEqual(len(tracker._packet_cohorts[0]), 0)
        self.assertEqual(tracker._pending_deadline_drop_counts.item(), 1.0)
        self.assertEqual(tracker.queue_states.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
