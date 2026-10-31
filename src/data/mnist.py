"""MNIST dataset wrapper that encodes images into spike trains."""
from __future__ import annotations

import torch
from torchvision import datasets, transforms

from src.data.base    import BaseDVSDataset
from src.coding.base  import BaseEncoder


class MNISTDataset(BaseDVSDataset):
    """Standard MNIST loaded via torchvision, encoded to spike trains on-the-fly."""

    _SENSOR_SIZE = (28, 28, 1)
    _N_CLASSES   = 10

    def __init__(
        self,
        encoder:    BaseEncoder,
        time_steps: int  = 20,
        data_dir:   str  = "data/",
        train:      bool = True,
    ):
        self.encoder    = encoder
        self.time_steps = time_steps

        ds = datasets.MNIST(
            root      = data_dir,
            train     = train,
            download  = True,
            transform = transforms.ToTensor(),
        )
        self._images = ds.data.float().div_(255.0).view(-1, 784)
        self._labels = ds.targets

    @property
    def sensor_size(self) -> tuple:
        return self._SENSOR_SIZE

    @property
    def n_classes(self) -> int:
        return self._N_CLASSES

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx):
        """Return the encoded spike train [T, D] and the label of one sample."""
        x      = self._images[idx].unsqueeze(0)
        spikes = self.encoder(x, self.time_steps)
        return spikes[:, 0, :], self._labels[idx]
