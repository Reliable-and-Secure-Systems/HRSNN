from __future__ import annotations
import torch
from src.coding.base import BaseEncoder


class PopulationEncoder(BaseEncoder):
    """Population (Gaussian receptive field) coding."""

    def __init__(self, n_gaussians: int = 10, sigma: float | None = None):
        self.n_gaussians = n_gaussians
        self.sigma = sigma if sigma is not None else 1.0 / max(1, n_gaussians - 1)

    def encode(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode each input with K Gaussian receptive fields, rate-coded; returns [T, B, N*K]."""
        B, N = x.shape
        centers    = torch.linspace(0.0, 1.0, self.n_gaussians, device=x.device)
        activation = torch.exp(
            -0.5 * ((x.unsqueeze(-1) - centers) / self.sigma) ** 2
        ).view(B, N * self.n_gaussians)

        expanded = activation.unsqueeze(0).expand(time_steps, -1, -1)
        return torch.bernoulli(expanded.clamp(0.0, 1.0))

    def output_dim(self, input_dim: int) -> int:
        return input_dim * self.n_gaussians
