from __future__ import annotations
import torch
from src.coding.base import BaseEncoder


class RateEncoder(BaseEncoder):
    """Bernoulli/Poisson rate coding."""

    def encode(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode each input as Bernoulli spikes with probability equal to its value; returns [T, B, N]."""
        x_clamped = x.clamp(0.0, 1.0)
        expanded  = x_clamped.unsqueeze(0).expand(time_steps, -1, -1)
        return torch.bernoulli(expanded)
