import unittest

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


if __name__ == "__main__":
    unittest.main()
