from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDVSDataset(Dataset):
    """Poisson spike trains shaped like a DVS dataset. Returns (spikes [T, N], label)."""

    def __init__(
        self,
        n_samples: int    = 1000,
        time_steps: int   = 20,
        input_dim: int    = 2312,
        n_classes: int    = 10,
        firing_rate: float = 0.1,
        seed: int         = 42,
    ):
        rng         = np.random.default_rng(seed)
        self.spikes = torch.from_numpy(
            rng.binomial(1, firing_rate,
                         (n_samples, time_steps, input_dim)).astype(np.float32)
        )
        self.labels = torch.from_numpy(
            rng.integers(0, n_classes, n_samples).astype(np.int64)
        )
        self._n_classes = n_classes

    @property
    def n_classes(self):
        return self._n_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spikes[idx], self.labels[idx]
