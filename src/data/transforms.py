from __future__ import annotations
import torch
import numpy as np
from typing import Tuple


Sample = Tuple[torch.Tensor, int]


class FlattenSpikes:
    """[T, C, H, W] numpy array → [T, C*H*W] binary float tensor. Applied after tonic ToFrame transform."""

    def __call__(self, sample: Sample) -> Sample:
        frames, label = sample
        frames = torch.as_tensor(frames, dtype=torch.float32)
        frames = (frames > 0).float()
        T, C, H, W = frames.shape
        return frames.view(T, C * H * W), label


class BinarizeSpikes:
    """Clamp any positive count to 1 (already binary after ToFrame, but explicit)."""

    def __call__(self, sample: Sample) -> Sample:
        frames, label = sample
        return (frames > 0).float(), label


class TemporalSubsample:
    """Uniformly subsample T time bins down to n_steps. Selects evenly spaced indices."""

    def __init__(self, n_steps: int):
        self.n_steps = n_steps

    def __call__(self, sample: Sample) -> Sample:
        frames, label = sample
        T = frames.shape[0]
        idx = torch.linspace(0, T - 1, self.n_steps).long()
        return frames[idx], label


class SpatialDownsample:
    """Average-pool spatial dimensions by an integer factor. Operates on [T, C, H, W] tensors."""

    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, sample: Sample) -> Sample:
        frames, label = sample
        if self.factor == 1:
            return frames, label
        frames = torch.as_tensor(frames, dtype=torch.float32)
        T, C, H, W = frames.shape
        frames = frames.view(T * C, 1, H, W)
        frames = torch.nn.functional.avg_pool2d(
            frames, kernel_size=self.factor, stride=self.factor
        )
        _, _, nH, nW = frames.shape
        frames = frames.view(T, C, nH, nW)
        return frames, label


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample
