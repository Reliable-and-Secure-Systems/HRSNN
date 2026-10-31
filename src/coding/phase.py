from __future__ import annotations
import torch
from src.coding.base import BaseEncoder


class PhaseEncoder(BaseEncoder):
    """Phase coding."""

    def __init__(self, n_periods: int = 4, threshold: float = 0.0):
        self.n_periods = n_periods
        self.threshold = threshold

    def encode(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode each input as one spike per oscillation period, its phase set by the value; returns [T, B, N]."""
        period_len = max(1, time_steps // self.n_periods)
        has_spike  = (x > self.threshold).float()
        offset     = ((1.0 - x) * (period_len - 1)).long().clamp(0, period_len - 1)

        spikes = torch.zeros(time_steps, x.size(0), x.size(1), device=x.device)
        for p in range(self.n_periods):
            fire_t = (p * period_len + offset).clamp(0, time_steps - 1)
            t_range = torch.arange(time_steps, device=x.device).view(time_steps, 1, 1)
            spikes += (t_range == fire_t.unsqueeze(0)).float()

        return (spikes * has_spike.unsqueeze(0)).clamp(0.0, 1.0)
