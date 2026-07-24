import unittest

import numpy as np

from baseline.spatial_dl_adapter import project_3d_links_to_spatial_dl


class SpatialDLAdapterTest(unittest.TestCase):
    def test_xy_projection_mask_and_link_roundtrip(self):
        positions = np.array(
            [[10, 20, 50], [30, 40, 150], [400, 20, 80]], dtype=float
        )
        links = np.array([[0, 1], [2, 0]], dtype=int)
        layout, mask, mapping = project_3d_links_to_spatial_dl(
            positions, links, target_link_count=50
        )
        self.assertEqual(layout.shape, (50, 4))
        self.assertEqual(mask.sum(), 2)
        self.assertTrue(np.all(mask[2:] == 0))
        self.assertEqual(mapping.scope, "full_link_set")
        project_action = np.array([1, 0], dtype=np.float32)
        legacy_action = mapping.project_to_legacy(project_action)
        np.testing.assert_array_equal(
            mapping.legacy_to_project(legacy_action), project_action
        )

    def test_more_than_50_links_uses_full_link_set(self):
        positions = np.column_stack(
            [np.arange(61), np.arange(61) * 2, np.ones(61) * 100]
        )
        links = np.array([[i, i + 1] for i in range(60)], dtype=int)
        layout, mask, mapping = project_3d_links_to_spatial_dl(positions, links)
        self.assertEqual(layout.shape, (60, 4))
        self.assertEqual(mask.sum(), 60)
        self.assertEqual(mapping.scope, "full_link_set")
        selected = np.arange(50)
        layout, mask, mapping = project_3d_links_to_spatial_dl(
            positions,
            links,
            common_subproblem_indices=selected,
            target_link_count=55,
        )
        self.assertEqual(layout.shape, (55, 4))
        self.assertEqual(mask.sum(), 50)
        np.testing.assert_array_equal(mapping.selected_project_indices, selected)
        self.assertEqual(mapping.scope, "explicit_common_subproblem")

    def test_target_length_never_crops_real_links(self):
        positions = np.column_stack(
            [np.arange(6), np.arange(6), np.ones(6) * 100]
        )
        links = np.array([[i, i + 1] for i in range(5)], dtype=int)
        with self.assertRaisesRegex(ValueError, "never cropped"):
            project_3d_links_to_spatial_dl(
                positions, links, target_link_count=4
            )


if __name__ == "__main__":
    unittest.main()
