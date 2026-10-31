from __future__ import annotations
import torch
from src.coding.base import BaseEncoder


class BurstEncoder(BaseEncoder):
    """Burst coding."""

    def __init__(self, max_burst: int = 10):
        self.max_burst = max_burst

    def encode(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode each input as consecutive spikes from t = 0, more for higher values; returns [T, B, N]."""
        n_spikes = (x.clamp(0.0, 1.0) * self.max_burst).round().long()
        n_burst  = min(self.max_burst, time_steps)

        spikes = torch.zeros(time_steps, x.size(0), x.size(1), device=x.device)
        for t in range(n_burst):
            spikes[t] = (n_spikes > t).float()

        return spikes
