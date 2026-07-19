import unittest
from dataclasses import replace

import torch

from simulator.mobile_block_fading import (
    build_observed_network_state,
    build_network_state_from_power,
    csi_error_db,
    generate_mobile_block_fading_trace,
    message_overhead_audit,
    run_mobile_queue_simulation,
)
from utils.project_setup import build_network_generator, load_config


class FirstNonemptyScheduler:
    def schedule_with_state(self, state, queues):
        action = torch.zeros(state.num_vertices, 1)
        nonempty = torch.where(queues > 0)[0]
        if nonempty.numel() > 0:
            action[nonempty[0], 0] = 1.0
        return action


class MobileBlockFadingTest(unittest.TestCase):
    def setUp(self):
        cfg = load_config("configs/paper.yaml")
        cfg.update({"NUM_UAV": 5, "TX_POWER": 10.0, "CSI_MODE": "perfect"})
        self.net_gen = build_network_generator(cfg)
        self.trace = generate_mobile_block_fading_trace(
            self.net_gen,
            num_slots=12,
            slots_per_block=4,
            slot_duration=0.001,
            speed_mps=10.0,
            seed=7,
            quantization_bits=(2, 4),
        )

    def test_persistent_link_identity_is_unique_and_sorted(self):
        pairs = [tuple(pair) for pair in self.trace.links.tolist()]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(pairs, sorted(pairs))
        self.assertEqual(len(self.trace.blocks), 3)
        for block in self.trace.blocks:
            self.assertEqual(block.available.numel(), self.trace.num_links)

    def test_observed_subset_maps_back_to_persistent_ids(self):
        state, persistent = build_observed_network_state(
            self.net_gen, self.trace, 1, "large_scale", q_bits=4
        )
        self.assertTrue(torch.equal(state.links, self.trace.links[persistent]))
        self.assertTrue(torch.all(persistent[1:] > persistent[:-1]))

    def test_latest_has_zero_csi_error_and_stale_uses_previous_block(self):
        self.assertAlmostEqual(
            csi_error_db(self.net_gen, self.trace, 1, "latest", 4), 0.0, places=6
        )
        latest0, _ = build_observed_network_state(
            self.net_gen, self.trace, 0, "latest", 4
        )
        stale1, _ = build_observed_network_state(
            self.net_gen, self.trace, 1, "stale", 4
        )
        self.assertTrue(torch.equal(latest0.signal_power, stale1.signal_power))

    def test_stale_age_is_independent_of_channel_block_length(self):
        latest0, _ = build_observed_network_state(
            self.net_gen, self.trace, 0, "latest", 4
        )
        stale2, _ = build_observed_network_state(
            self.net_gen, self.trace, 2, "stale", 4, stale_blocks=2
        )
        self.assertTrue(torch.equal(latest0.signal_power, stale2.signal_power))

    def test_trace_retains_complex_channel_consistent_with_power(self):
        for block in self.trace.blocks:
            reconstructed = self.net_gen.tx_power * block.true_channel.abs().square()
            self.assertTrue(torch.allclose(
                reconstructed, block.true_power, atol=1e-12, rtol=1e-6
            ))

    def test_predicted_power_builder_uses_one_coherent_matrix(self):
        predicted = self.trace.blocks[1].large_scale_power * 1.1
        state, persistent = build_network_state_from_power(
            self.net_gen, self.trace, 1, predicted, csi_label="unit_test"
        )
        self.assertEqual(state.csi_mode, "unit_test")
        self.assertTrue(torch.equal(state.links, self.trace.links[persistent]))
        self.assertTrue(torch.allclose(state.signal_power, predicted))

    def test_rician_variants_preserve_pairing_and_pure_los_power(self):
        kwargs = dict(
            num_slots=12,
            slots_per_block=4,
            slot_duration=0.001,
            speed_mps=10.0,
            seed=17,
            quantization_bits=(4,),
        )
        rayleigh = generate_mobile_block_fading_trace(
            self.net_gen, **kwargs
        )
        rician = generate_mobile_block_fading_trace(
            self.net_gen, **kwargs, rician_k_db=5.0
        )
        pure_los = generate_mobile_block_fading_trace(
            self.net_gen, **kwargs, rician_k_db=float("inf")
        )
        self.assertTrue(torch.equal(rayleigh.links, rician.links))
        self.assertTrue(torch.equal(rayleigh.links, pure_los.links))
        for rayleigh_block, rician_block, los_block in zip(
            rayleigh.blocks, rician.blocks, pure_los.blocks
        ):
            self.assertEqual(rayleigh_block.available.tolist(), rician_block.available.tolist())
            self.assertTrue(torch.allclose(
                los_block.true_power, los_block.large_scale_power,
                atol=1e-12, rtol=1e-6,
            ))

    def test_fixed_generator_uses_normalized_rician_off_diagonal_power(self):
        self.assertEqual(self.net_gen.channel_model, "rician")
        torch.manual_seed(123)
        positions = self.net_gen._init_positions()
        normalized_power = []
        for _ in range(512):
            _, h, _, large_scale = self.net_gen._channel_gain(positions)
            mask = large_scale > 0
            normalized_power.append(
                (h.abs().square()[mask] / large_scale.square()[mask]).mean()
            )
        self.assertTrue(torch.isfinite(large_scale).all())
        self.assertTrue(torch.equal(
            torch.diagonal(large_scale), torch.zeros(self.net_gen.num_uavs)
        ))
        self.assertAlmostEqual(
            float(torch.stack(normalized_power).mean()), 1.0, delta=0.05
        )

    def test_queue_simulation_keeps_persistent_dimension(self):
        arrivals = torch.zeros(self.trace.num_slots, self.trace.num_links)
        arrivals[:, 0] = 1.0
        metrics = run_mobile_queue_simulation(
            self.net_gen,
            FirstNonemptyScheduler(),
            self.trace,
            arrivals,
            csi_mode="latest",
        )
        self.assertEqual(metrics["num_persistent_links"], self.trace.num_links)
        self.assertGreaterEqual(metrics["total_departures"], 0.0)
        self.assertLessEqual(metrics["total_departures"], metrics["total_arrivals"])
        self.assertEqual(metrics["packet_count_conservation_error"], 0.0)
        self.assertEqual(
            metrics["delivered_packet_count"] + metrics["unfinished_packet_count"],
            int(metrics["total_arrivals"]),
        )
        self.assertLess(metrics["packet_work_conservation_error"], 1e-4)

    def test_temporarily_unavailable_link_retains_arrivals(self):
        unavailable_blocks = tuple(
            replace(block, available=torch.zeros_like(block.available))
            for block in self.trace.blocks
        )
        unavailable_trace = replace(self.trace, blocks=unavailable_blocks)
        arrivals = torch.zeros(self.trace.num_slots, self.trace.num_links)
        arrivals[:, 0] = 1.0
        metrics = run_mobile_queue_simulation(
            self.net_gen,
            FirstNonemptyScheduler(),
            unavailable_trace,
            arrivals,
            csi_mode="latest",
        )
        self.assertEqual(metrics["total_departures"], 0.0)
        self.assertEqual(metrics["final_total_backlog"], float(self.trace.num_slots))
        self.assertEqual(metrics["delivered_packet_count"], 0)
        self.assertEqual(metrics["unfinished_packet_count"], self.trace.num_slots)
        self.assertEqual(metrics["packet_count_conservation_error"], 0.0)

    def test_pure_omits_channel_gat_message_rounds(self):
        gat = message_overhead_audit(
            20, 5, 100, slots_per_block=5, uses_channel_gat=True
        )
        pure = message_overhead_audit(
            20, 5, 100, slots_per_block=5, uses_channel_gat=False
        )
        self.assertEqual(gat["channel_rounds_per_block"], 5)
        self.assertEqual(pure["channel_rounds_per_block"], 0)
        self.assertEqual(gat["amortized_algorithm_rounds_per_slot"], 4.0)
        self.assertEqual(pure["amortized_algorithm_rounds_per_slot"], 3.0)
        self.assertGreater(
            gat["link_edge_unicast_bits_per_slot"],
            pure["link_edge_unicast_bits_per_slot"],
        )


if __name__ == "__main__":
    unittest.main()
