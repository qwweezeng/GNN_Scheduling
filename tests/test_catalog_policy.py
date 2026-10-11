import unittest
from pathlib import Path

import torch

from utils.artifacts import (
    expected_dataset_metadata_quantized,
    paper_dataset_paths_quantized,
)
from utils.project_setup import build_network_generator, load_config


class CatalogPolicyTest(unittest.TestCase):
    def _generator(self):
        config = load_config("configs/paper.yaml")
        config.update({
            "NUM_UAV": 12,
            "CHANNEL_MODEL": "rician",
            "RICIAN_K_DB": 5.0,
            "CSI_MODE": "perfect",
        })
        return config, build_network_generator(config)

    def test_paired_policies_change_only_catalog_derived_graph(self):
        _, generator = self._generator()
        torch.manual_seed(2026)
        instantaneous = generator.generate_network(
            csi_mode="perfect", include_queue_feature=False,
            catalog_policy="instantaneous",
        )
        torch.manual_seed(2026)
        mean_power = generator.generate_network(
            csi_mode="perfect", include_queue_feature=False,
            catalog_policy="mean_power",
        )

        self.assertTrue(torch.equal(instantaneous.positions, mean_power.positions))
        self.assertTrue(torch.equal(instantaneous.signal_power, mean_power.signal_power))
        self.assertTrue(torch.equal(
            instantaneous.signal_power_large_scale,
            mean_power.signal_power_large_scale,
        ))
        self.assertEqual(instantaneous.catalog_policy, "instantaneous")
        self.assertEqual(mean_power.catalog_policy, "mean_power")

        inst_links = {tuple(row) for row in instantaneous.links.tolist()}
        mean_links = {tuple(row) for row in mean_power.links.tolist()}
        self.assertNotEqual(inst_links, mean_links)
        shared = inst_links & mean_links
        self.assertTrue(shared)
        inst_index = {tuple(row): i for i, row in enumerate(instantaneous.links.tolist())}
        mean_index = {tuple(row): i for i, row in enumerate(mean_power.links.tolist())}
        for link in shared:
            self.assertAlmostEqual(
                float(instantaneous.vertex_features[inst_index[link], 0]),
                float(mean_power.vertex_features[mean_index[link], 0]),
                places=6,
            )

    def test_default_perfect_policy_preserves_legacy_behavior(self):
        _, generator = self._generator()
        torch.manual_seed(77)
        default = generator.generate_network(
            csi_mode="perfect", include_queue_feature=False,
        )
        torch.manual_seed(77)
        explicit = generator.generate_network(
            csi_mode="perfect", include_queue_feature=False,
            catalog_policy="instantaneous",
        )
        self.assertTrue(torch.equal(default.links, explicit.links))
        self.assertTrue(torch.equal(default.vertex_features, explicit.vertex_features))
        self.assertTrue(torch.equal(default.edge_index, explicit.edge_index))
        self.assertTrue(torch.equal(default.edge_weights, explicit.edge_weights))

    def test_explicit_policy_has_distinct_path_and_fingerprint(self):
        config, _ = self._generator()
        seeds = {"train": 10086, "val": 10087, "test": 10088}
        sizes = {"train": 300, "val": 128, "test": 128}
        legacy = expected_dataset_metadata_quantized(
            config, 12, seeds, sizes, "perfect", None,
        )
        instantaneous = expected_dataset_metadata_quantized(
            config, 12, seeds, sizes, "perfect", None,
            catalog_policy="instantaneous",
        )
        mean_power = expected_dataset_metadata_quantized(
            config, 12, seeds, sizes, "perfect", None,
            catalog_policy="mean_power",
        )
        self.assertNotIn("catalog_policy", legacy["physics"])
        self.assertEqual(
            instantaneous["physics"]["catalog_policy"], "instantaneous"
        )
        self.assertEqual(mean_power["physics"]["catalog_policy"], "mean_power")
        self.assertEqual(len({
            legacy["fingerprint"], instantaneous["fingerprint"],
            mean_power["fingerprint"],
        }), 3)

        root = Path("datasets/catalog_pilot")
        instant_paths = paper_dataset_paths_quantized(
            root, 12, seeds, sizes, "perfect", None,
            catalog_policy="instantaneous",
        )
        mean_paths = paper_dataset_paths_quantized(
            root, 12, seeds, sizes, "perfect", None,
            catalog_policy="mean_power",
        )
        self.assertNotEqual(instant_paths, mean_paths)
        self.assertIn("catalog_instantaneous", str(instant_paths[0]))
        self.assertIn("catalog_mean_power", str(mean_paths[0]))


if __name__ == "__main__":
    unittest.main()
