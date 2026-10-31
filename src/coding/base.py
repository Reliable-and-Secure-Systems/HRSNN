from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class BaseEncoder(ABC):
    """Contract for all spike coding schemes."""

    @abstractmethod
    def encode(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode inputs [B, N] in [0, 1] into a spike train [T, B, N]."""

    def output_dim(self, input_dim: int) -> int:
        """Effective feature dimension after encoding."""
        return input_dim

    def __call__(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        assert x.dim() == 2, f"Expected [B, N], got shape {x.shape}"
        return self.encode(x, time_steps)
