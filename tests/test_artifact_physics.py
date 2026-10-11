import unittest

from utils.artifacts import (
    dataset_split_fingerprint,
    expected_dataset_metadata,
    validate_dataset_payload,
)
from utils.project_setup import (
    channel_checkpoint_path,
    physics_tag,
    queue_checkpoint_path,
    queue_training_cache_path,
)


class ArtifactPhysicsIdentityTest(unittest.TestCase):
    def test_tx_power_and_csi_are_part_of_artifact_identity(self):
        tx10 = {"TX_POWER": 10, "CSI_MODE": "perfect"}
        tx22 = {"TX_POWER": 22, "CSI_MODE": "perfect"}
        self.assertEqual(physics_tag(tx10), "tx=10dBm_csi=perfect")
        self.assertNotEqual(
            queue_training_cache_path(tx10, 10),
            queue_training_cache_path(tx22, 10),
        )
        self.assertNotEqual(
            queue_checkpoint_path(tx10, 10),
            queue_checkpoint_path(tx22, 10),
        )

    def test_queue_and_channel_datasets_remain_separate(self):
        cfg = {"TX_POWER": 10, "CSI_MODE": "perfect"}
        channel = str(channel_checkpoint_path(cfg, 20))
        queue = str(queue_checkpoint_path(cfg, 20))
        self.assertIn("tx=10dBm_csi=perfect", channel)
        self.assertIn("tx=10dBm_csi=perfect", queue)
        self.assertNotEqual(channel, queue)

    def test_test_split_identity_is_independent_of_training_pool_size(self):
        config = {
            "AREA_WIDTH": 1000,
            "HEIGHT_RANGE": [50, 150],
            "MIN_DISTANCE": 20,
            "TX_POWER": 10,
            "BANDWIDTH": 5e6,
            "CARRIER_FREQ": 2.4e9,
            "NOISE_PSD": -169,
            "SNR_THRES": 5,
            "SNR_GAP_DB": 0,
            "CHANNEL_MODEL": "rician",
            "RICIAN_K_DB": 5,
            "CSI_MODE": "perfect",
        }
        seeds = {"train": 10086, "val": 10087, "test": 10088}
        metadata_300 = expected_dataset_metadata(
            config, 20, seeds, {"train": 300, "val": 128, "test": 128}
        )
        metadata_500 = expected_dataset_metadata(
            config, 20, seeds, {"train": 500, "val": 128, "test": 128}
        )

        self.assertNotEqual(metadata_300["fingerprint"], metadata_500["fingerprint"])
        self.assertEqual(
            dataset_split_fingerprint(metadata_300, "test"),
            dataset_split_fingerprint(metadata_500, "test"),
        )
        payload = {"metadata": metadata_500, "test_set": list(range(128))}
        self.assertEqual(
            validate_dataset_payload(payload, metadata_300, ("test",)), []
        )
        self.assertIn(
            "metadata.split_sizes.train mismatch",
            validate_dataset_payload(
                {"metadata": metadata_500, "train_set": list(range(500))},
                metadata_300,
                ("train",),
            ),
        )


if __name__ == "__main__":
    unittest.main()
