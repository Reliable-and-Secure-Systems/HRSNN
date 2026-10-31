from __future__ import annotations
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDVSDataset(Dataset, ABC):
    """Contract for all datasets."""

    @property
    @abstractmethod
    def sensor_size(self) -> tuple:
        """Return (H, W, C) of the sensor."""

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of output classes."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return (spikes [T, N], label) for sample at idx."""
        pass
