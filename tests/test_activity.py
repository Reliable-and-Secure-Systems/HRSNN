"""Spike and SynOps accounting (src/tracking/activity.py) on a model whose answer can be worked out by hand."""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.tracking.activity import measure_activity


class _Readout(nn.Module):
    """One identity Linear layer applied per time step: x [T, B, 2] -> [T, B, 2]."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))

    def forward(self, x):
        out = torch.stack([self.fc(x[t]) for t in range(x.shape[0])])
        return out, out


class MeasureActivityTest(unittest.TestCase):

    def setUp(self):
        a = [[1, 0], [1, 0], [1, 0]]
        b = [[0, 1], [1, 0], [1, 0]]
        c = [[0, 1], [0, 1], [0, 1]]
        spikes = torch.tensor([a, b, c], dtype=torch.float32)
        labels = torch.tensor([0, 0, 0])
        self.loader = [(spikes[:2], labels[:2]), (spikes[2:], labels[2:])]
        self.act = measure_activity(_Readout(), self.loader, torch.device("cpu"))

    def test_counts(self):
        act = self.act
        self.assertEqual(act["n_samples"], 3)
        self.assertEqual(act["time_steps"], 3)
        self.assertAlmostEqual(act["spikes_per_sample"], 9 / 3)
        self.assertAlmostEqual(act["mean_spike_rate"], 9 / (3 * 3 * 2))
        self.assertAlmostEqual(act["syn_ops_per_sample"], 9 * 2 / 3)
        self.assertAlmostEqual(act["ann_macs"], 4.0)
        self.assertAlmostEqual(act["efficiency_ratio"], (18 / 3) / 4)

    def test_per_layer(self):
        (layer,) = self.act["layers"]
        self.assertEqual((layer["name"], layer["fan_in"], layer["fan_out"]), ("fc", 2, 2))
        self.assertAlmostEqual(layer["syn_ops_per_sample"], self.act["syn_ops_per_sample"])

    def test_accuracy_by_timestep(self):
        for got, want in zip(self.act["acc_by_timestep"], [1 / 3, 2 / 3, 2 / 3]):
            self.assertAlmostEqual(got, want)

    def test_decision_step_histogram(self):
        self.assertEqual(self.act["decision_step_hist"], [1, 1, 0])

    def test_max_batches(self):
        act = measure_activity(_Readout(), self.loader, torch.device("cpu"), max_batches=1)
        self.assertEqual(act["n_samples"], 2)

    def test_hooks_removed(self):
        model = _Readout()
        measure_activity(model, self.loader, torch.device("cpu"))
        self.assertEqual(len(model.fc._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
