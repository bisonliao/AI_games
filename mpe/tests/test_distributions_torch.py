import unittest

import numpy as np
import torch
from gymnasium import spaces

from maddpg.common.distributions_torch import (
    SoftCategoricalPd,
    SoftMultiCategoricalPdType,
    make_pdtype,
)


class TorchDistributionParityTest(unittest.TestCase):
    def test_soft_categorical_matches_fixed_gumbel_formula_and_has_gradient(self):
        logits = torch.tensor([[0.2, -0.1, 0.7]], requires_grad=True)
        uniform = torch.tensor([[0.2, 0.5, 0.8]])
        sample = SoftCategoricalPd(logits).sample(uniform=uniform)
        expected = torch.softmax(logits - torch.log(-torch.log(uniform)), dim=-1)

        torch.testing.assert_close(sample, expected)
        torch.testing.assert_close(sample.sum(-1), torch.ones(1))
        (sample[:, 0] - sample[:, 1]).sum().backward()
        self.assertGreater(torch.linalg.vector_norm(logits.grad).item(), 0.0)

    def test_multi_categorical_preserves_independent_branches(self):
        pdtype = SoftMultiCategoricalPdType((5, 3))
        logits = torch.zeros((2, 8), requires_grad=True)
        sample = pdtype.pdfromflat(logits).sample()

        self.assertEqual(tuple(sample.shape), (2, 8))
        torch.testing.assert_close(sample[:, :5].sum(-1), torch.ones(2))
        torch.testing.assert_close(sample[:, 5:].sum(-1), torch.ones(2))
        sample[:, 1].sum().backward()
        self.assertGreater(torch.linalg.vector_norm(logits.grad).item(), 0.0)

    def test_gymnasium_multidiscrete_uses_sum_not_product(self):
        pdtype = make_pdtype(spaces.MultiDiscrete(np.array([5, 3])))
        self.assertEqual(pdtype.param_shape(), [8])
        self.assertEqual(pdtype.sample_shape(), [8])


if __name__ == "__main__":
    unittest.main()
