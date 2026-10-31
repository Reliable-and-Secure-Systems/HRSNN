from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLearner(ABC):
    """Contract for all learning algorithms."""

    @abstractmethod
    def setup(self, model: nn.Module, cfg, device: torch.device):
        """Initialize optimizer, scheduler, AMP scaler, any internal state."""

    @abstractmethod
    def step(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
    ) -> tuple[float, torch.Tensor]:
        """One update step."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, val_acc: float):
        """Called at end of each epoch. Use for LR scheduling etc."""

    @property
    def current_lr(self) -> float:
        """Current learning rate (0.0 for non-gradient learners)."""
        return 0.0
