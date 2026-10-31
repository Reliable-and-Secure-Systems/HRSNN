"""DeviceEncodedLoader must yield what the per-image dataset path yields: the same layout, and for a deterministic encoder the same spikes."""
from __future__ import annotations

import unittest

import torch

from src.coding.registry import get_encoder
from src.data.loaders import DeviceEncodedLoader


class DeviceEncodedLoaderTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.images = torch.rand(10, 6)
        self.labels = torch.arange(10)

    def _per_image(self, encoder, T):
        """What MNISTDataset / CIFAR10Dataset.__getitem__ return, stacked: [N, T, D]."""
        return torch.stack([encoder(self.images[i].unsqueeze(0), T)[:, 0, :]
                            for i in range(len(self.labels))])

    def _batched(self, encoder, T, batch_size=4):
        loader = DeviceEncodedLoader(self.images, self.labels, encoder, T,
                                     batch_size, torch.device("cpu"))
        batches = list(loader)
        self.assertEqual(len(batches), len(loader))
        spikes = torch.cat([b[0] for b in batches])
        labels = torch.cat([b[1] for b in batches])
        return spikes, labels

    def test_deterministic_encoders_match_exactly(self):
        for name in ("ttfs", "phase", "burst"):
            enc = get_encoder(name)
            spikes, labels = self._batched(enc, 20)
            self.assertTrue(torch.equal(spikes, self._per_image(enc, 20)), name)
            self.assertTrue(torch.equal(labels, self.labels))

    def test_layout_and_population_width(self):
        enc = get_encoder("population", n_gaussians=10)
        spikes, _ = self._batched(enc, 20)
        self.assertEqual(tuple(spikes.shape), (10, 20, 60))

    def test_stochastic_encoder_reproducible_under_seed(self):
        enc = get_encoder("rate")
        torch.manual_seed(7)
        a, _ = self._batched(enc, 20)
        torch.manual_seed(7)
        b, _ = self._batched(enc, 20)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
