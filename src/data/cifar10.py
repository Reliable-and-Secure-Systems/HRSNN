"""CIFAR-10 dataset wrapper that encodes images into spike trains."""
from __future__ import annotations

import torch
from torchvision import datasets, transforms

from src.data.base   import BaseDVSDataset
from src.coding.base import BaseEncoder


class CIFAR10Dataset(BaseDVSDataset):
    """Standard CIFAR-10 loaded via torchvision, encoded to spike trains on-the-fly."""

    _SENSOR_SIZE = (32, 32, 3)
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

        ds = datasets.CIFAR10(
            root      = data_dir,
            train     = train,
            download  = True,
            transform = transforms.ToTensor(),
        )
        images = torch.tensor(ds.data, dtype=torch.float32).div_(255.0)
        self._images = images.permute(0, 3, 1, 2).reshape(-1, 3072)
        self._labels = torch.tensor(ds.targets, dtype=torch.int64)

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
