from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.data.synthetic import SyntheticDVSDataset


def build_dataloaders(
    dataset_train,
    dataset_test,
    batch_size: int   = 64,
    val_split: float  = 0.1,
    num_workers: int  = 4,
    pin_memory: bool  = True,
    seed: int         = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset_train into train/val, return three DataLoaders."""
    n_val   = int(len(dataset_train) * val_split)
    n_train = len(dataset_train) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset_train, [n_train, n_val], generator=gen)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (
        DataLoader(train_ds,    shuffle=True,  **kw),
        DataLoader(val_ds,      shuffle=False, **kw),
        DataLoader(dataset_test, shuffle=False, **kw),
    )


class DeviceEncodedLoader:
    """Test-set loader that encodes spikes on the target device, a batch at a time."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, encoder,
                 time_steps: int, batch_size: int, device: torch.device):
        self.images     = images
        self.labels     = labels
        self.encoder    = encoder
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.device     = device

    def __len__(self) -> int:
        return -(-len(self.labels) // self.batch_size)

    def __iter__(self):
        for start in range(0, len(self.labels), self.batch_size):
            x = self.images[start:start + self.batch_size].to(self.device, non_blocking=True)
            y = self.labels[start:start + self.batch_size].to(self.device, non_blocking=True)
            spikes = self.encoder(x, self.time_steps)
            yield spikes.permute(1, 0, 2), y


def build_synthetic_loaders(
    time_steps: int  = 20,
    input_dim: int   = 2312,
    n_classes: int   = 10,
    batch_size: int  = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Synthetic loaders for pipeline testing. No download required."""
    train_ds = SyntheticDVSDataset(5000, time_steps, input_dim, n_classes, seed=0)
    val_ds   = SyntheticDVSDataset(1000, time_steps, input_dim, n_classes, seed=1)
    test_ds  = SyntheticDVSDataset(1000, time_steps, input_dim, n_classes, seed=2)

    kw = dict(batch_size=batch_size, num_workers=0)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )
