from __future__ import annotations
import torch
from src.coding.base import BaseEncoder


class TTFSEncoder(BaseEncoder):
    """Time-To-First-Spike (TTFS) coding."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def encode(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode each input as at most one spike, earlier for higher values; returns [T, B, N]."""
        spike_t = ((1.0 - x) * (time_steps - 1)).long().clamp(0, time_steps - 1)
        has_spike = (x > self.threshold).float()

        t_range = torch.arange(time_steps, device=x.device).view(time_steps, 1, 1)
        spikes  = (t_range == spike_t.unsqueeze(0)).float()
        return spikes * has_spike.unsqueeze(0)
