import unittest

import torch

from models.complex_geometry_csi import ComplexGeometryCSIPredictor


class ComplexGeometryCSIPredictorTest(unittest.TestCase):
    def test_formal_model_is_link_local_and_has_frozen_size(self):
        model = ComplexGeometryCSIPredictor(input_dim=10, hidden_dim=16)
        features = torch.randn(7, 10)
        prediction, hidden = model.forward_step(features)
        self.assertEqual(tuple(prediction.shape), (7,))
        self.assertIsNone(hidden)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 897)


if __name__ == "__main__":
    unittest.main()
