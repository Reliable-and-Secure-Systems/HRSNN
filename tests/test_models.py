"""Per-time-step BatchNorm (src/models/blocks.py: StepBatchNorm1d)."""
import unittest

import torch
import torch.nn as nn

from src.models.blocks import StepBatchNorm1d
from src.models.mlp import SNNMLP


def _with_plain_bn(model: SNNMLP) -> SNNMLP:
    """The same model with the pre-2026-10-08 BatchNorm1d (one running average)."""
    for block in model.blocks:
        old = nn.BatchNorm1d(block.bn.num_features)
        old.weight.data.copy_(block.bn.weight.data)
        old.bias.data.copy_(block.bn.bias.data)
        old.reset_step = lambda: None
        block.bn = old
    return model


class StepBatchNormTest(unittest.TestCase):

    def test_training_matches_batchnorm1d(self):
        torch.manual_seed(0)
        step, plain = StepBatchNorm1d(6, time_steps=3), nn.BatchNorm1d(6)
        for t in range(3):
            x = torch.randn(8, 6) * (t + 1) + t
            self.assertTrue(torch.equal(step(x), plain(x)))

    def test_each_step_keeps_its_own_statistics(self):
        bn = StepBatchNorm1d(4, time_steps=2, momentum=1.0)
        a, b = torch.randn(16, 4), torch.randn(16, 4) * 3 + 5
        bn(a); bn(b)
        self.assertTrue(torch.allclose(bn.running_mean[0], a.mean(0)))
        self.assertTrue(torch.allclose(bn.running_mean[1], b.mean(0)))
        bn.eval(); bn.reset_step()
        bn(a)
        self.assertTrue(torch.allclose(bn(b), (b - b.mean(0)) / (b.var(0) + bn.eps).sqrt(),
                                       atol=1e-5))

    def test_too_many_steps_raises(self):
        bn = StepBatchNorm1d(4, time_steps=2)
        bn(torch.randn(3, 4)); bn(torch.randn(3, 4))
        with self.assertRaises(RuntimeError):
            bn(torch.randn(3, 4))

    def test_model_training_step_is_unchanged(self):
        torch.manual_seed(0)
        new = SNNMLP(input_dim=20, hidden_dims=[16, 8], n_classes=3, time_steps=10)
        old = _with_plain_bn(SNNMLP(input_dim=20, hidden_dims=[16, 8], n_classes=3,
                                    time_steps=10))
        old.load_state_dict({k: v for k, v in new.state_dict().items()
                             if "running" not in k and "num_batches" not in k}, strict=False)
        x = (torch.rand(10, 4, 20) < 0.3).float()
        y = torch.tensor([0, 1, 2, 0])
        outs = []
        for m in (new, old):
            m.train()
            torch.manual_seed(1)
            out, _ = m(x)
            nn.functional.cross_entropy(out.sum(0), y).backward()
            outs.append((out, [p.grad for p in m.parameters()]))
        self.assertTrue(torch.equal(outs[0][0], outs[1][0]))
        for g_new, g_old in zip(outs[0][1], outs[1][1]):
            self.assertTrue(torch.equal(g_new, g_old))

    def test_model_runs_repeatedly_and_stores_per_step_buffers(self):
        model = SNNMLP(input_dim=20, hidden_dims=[16, 8], n_classes=3, time_steps=10).eval()
        x = (torch.rand(10, 4, 20) < 0.3).float()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x)[0], model(x)[0]))
        self.assertEqual(tuple(model.state_dict()["blocks.0.bn.running_mean"].shape), (10, 16))


if __name__ == "__main__":
    unittest.main()
