import unittest

import torch

from experiments.audit_link_catalog import compare_catalog_masks


class CatalogAuditTests(unittest.TestCase):
    def test_comparison_counts_both_kinds_of_disagreement(self):
        instantaneous = torch.tensor(
            [[False, True, False], [False, False, True], [True, False, False]]
        )
        mean_power = torch.tensor(
            [[False, True, True], [False, False, False], [True, False, False]]
        )

        result = compare_catalog_masks(instantaneous, mean_power)

        self.assertEqual(result["intersection_count"], 2)
        self.assertEqual(result["instantaneous_only_count"], 1)
        self.assertEqual(result["mean_only_count"], 1)
        self.assertEqual(result["union_count"], 4)
        self.assertEqual(result["jaccard"], 0.5)
        self.assertEqual(result["symmetric_difference_over_mean"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
