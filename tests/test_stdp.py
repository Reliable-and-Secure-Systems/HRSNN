"""STDP learner (src/learning/stdp.py): timing, competition, scaling, training step."""
import unittest

import torch

from src.learning.stdp import STDPLearner
from src.models.mlp import SNNMLP


def _learner(model, **cfg):
    lr = STDPLearner()
    lr.setup(model, cfg, torch.device("cpu"))
    return lr


class STDPRuleTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = SNNMLP(input_dim=2, hidden_dims=[4, 3], n_classes=2, time_steps=3)

    def test_input_before_output_potentiates_after_depresses(self):
        lr = _learner(self.model, k_winners=0.25)
        pre = torch.zeros(3, 1, 2)
        pre[0, 0, 0] = 1.0
        pre[2, 0, 1] = 1.0
        post = torch.zeros(3, 1, 4)
        post[1, 0, 0] = 1.0
        dw = lr.stdp_update(pre, post)
        self.assertGreater(dw[0, 0].item(), 0.0)
        self.assertLess(dw[0, 1].item(), 0.0)

    def test_equal_amplitudes_do_not_cancel(self):
        lr = _learner(self.model, k_winners=0.25, a_plus=0.01, a_minus=0.01)
        pre = torch.zeros(3, 1, 2)
        pre[0, 0, 0] = 1.0
        post = torch.zeros(3, 1, 4)
        post[1, 0, 0] = 1.0
        self.assertNotEqual(lr.stdp_update(pre, post)[0, 0].item(), 0.0)

    def test_only_winners_learn(self):
        lr = _learner(self.model, k_winners=0.25)
        pre = torch.ones(3, 1, 2)
        post = torch.zeros(3, 1, 4)
        post[:, 0, 2] = 1.0
        post[1, 0, 1] = 1.0
        dw = lr.stdp_update(pre, post)
        self.assertTrue(torch.any(dw[2] != 0))
        self.assertTrue(torch.all(dw[[0, 1, 3]] == 0))

    def test_row_norms_are_preserved(self):
        lr = _learner(self.model, k_winners=1.0, stdp_lr=10.0)
        before = self.model.blocks[0].fc.weight.norm(dim=1).clone()
        pre = (torch.rand(3, 5, 2) < 0.5).float()
        post = (torch.rand(3, 5, 4) < 0.5).float()
        lr.stdp_update(pre, post)
        after = self.model.blocks[0].fc.weight.norm(dim=1)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))


class STDPPhaseTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = SNNMLP(input_dim=20, hidden_dims=[16, 8], n_classes=3, time_steps=10)
        self.lr = _learner(self.model, k_winners=0.25, stdp_lr=0.1, stdp_epochs=2)
        self.x = (torch.rand(10, 6, 20) < 0.5).float()
        self.y = torch.tensor([0, 1, 2, 0, 1, 2])

    def _weights(self):
        return [self.model.blocks[i].fc.weight.detach().clone() for i in (0, 1)]

    def test_phase_one_moves_only_the_stdp_layer(self):
        w1, w2 = self._weights()
        self.lr.step(self.x, self.y, self.model)
        self.assertFalse(torch.equal(self.model.blocks[0].fc.weight, w1))
        self.assertTrue(torch.equal(self.model.blocks[1].fc.weight, w2))

    def test_phase_two_freezes_the_stdp_layer_and_backprops_the_rest(self):
        for epoch in range(2):
            self.lr.on_epoch_end(epoch + 1, 0.0)
        self.assertFalse(self.lr.in_stdp_phase)
        w1, w2 = self._weights()
        self.lr.step(self.x, self.y, self.model)
        self.assertTrue(torch.equal(self.model.blocks[0].fc.weight, w1))
        self.assertFalse(torch.equal(self.model.blocks[1].fc.weight, w2))
        self.assertIsNone(self.model.blocks[0].fc.weight.grad)


if __name__ == "__main__":
    unittest.main()
