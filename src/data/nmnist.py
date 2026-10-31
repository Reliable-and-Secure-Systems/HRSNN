"""N-MNIST dataset wrapper using tonic, with a single-file RAM cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tonic
import tonic.transforms as transforms

from src.data.base import BaseDVSDataset


class NMNISTDataset(BaseDVSDataset):
    """N-MNIST (DVS) dataset with a single-file cache for fast training."""

    _SENSOR_SIZE: tuple = tonic.datasets.NMNIST.sensor_size
    _N_CLASSES:   int   = 10

    def __init__(
        self,
        data_dir:  str  = "data/",
        train:     bool = True,
        time_steps: int = 20,
        cache_dir: str  = "data/nmnist_cache",
    ):
        self.time_steps = time_steps
        split      = "train" if train else "test"
        cache_file = Path(cache_dir) / f"nmnist_{split}_T{time_steps}.npz"

        if cache_file.exists():
            self._load_cache(cache_file)
        else:
            self._build_and_save(data_dir, train, time_steps, cache_file)


    def _load_cache(self, path: Path):
        """Load pre-built npz into RAM — called on every run after the first."""
        data          = np.load(path)
        self._spikes  = torch.from_numpy(data["spikes"])
        self._labels  = torch.from_numpy(data["labels"])

    def _build_and_save(self, data_dir, train, time_steps, cache_file: Path):
        """One-time build: iterate tonic, collect all frames into a single array, compress and save."""
        print(f"[NMNISTDataset] Building cache → {cache_file}")

        raw = tonic.datasets.NMNIST(
            save_to   = data_dir,
            train     = train,
            transform = transforms.ToFrame(
                sensor_size = self._SENSOR_SIZE,
                n_time_bins = time_steps,
            ),
        )

        n      = len(raw)
        n_flat = self._SENSOR_SIZE[0] * self._SENSOR_SIZE[1] * self._SENSOR_SIZE[2]

        spikes = np.empty((n, time_steps, n_flat), dtype=np.uint8)
        labels = np.empty(n, dtype=np.int64)

        for i, (frames, label) in enumerate(raw):
            spikes[i] = (frames > 0).astype(np.uint8).reshape(time_steps, -1)
            labels[i] = int(label)
            if (i + 1) % 10_000 == 0:
                print(f"  {i + 1:>6}/{n}")

        print(f"  {n}/{n}  done — saving …")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_file), spikes=spikes, labels=labels)
        print(f"  Saved to {cache_file}\n")

        self._spikes = torch.from_numpy(spikes)
        self._labels = torch.from_numpy(labels)


    @property
    def sensor_size(self) -> tuple:
        return self._SENSOR_SIZE

    @property
    def n_classes(self) -> int:
        return self._N_CLASSES

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx):
        """Return the event spike train [T, N] and the label of one sample."""
        return self._spikes[idx].float(), self._labels[idx]
