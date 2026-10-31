from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neurons import get_neuron

try:
    import snntorch as snn
    _HAS_SNNTORCH = True
except ImportError:
    _HAS_SNNTORCH = False


class StepBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d with running statistics kept per time step (shared affine)."""

    def __init__(self, num_features: int, time_steps: int, **kw):
        super().__init__(num_features, **kw)
        self.time_steps = time_steps
        self.register_buffer("running_mean", torch.zeros(time_steps, num_features))
        self.register_buffer("running_var",  torch.ones(time_steps, num_features))
        self.step = 0

    def reset_step(self) -> None:
        self.step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.step
        if t >= self.time_steps:
            raise RuntimeError(
                f"StepBatchNorm1d called for step {t + 1} but holds statistics for "
                f"{self.time_steps} steps; call reset_states() between sequences."
            )
        self.step += 1
        return F.batch_norm(
            x, self.running_mean[t], self.running_var[t], self.weight, self.bias,
            self.training, self.momentum, self.eps,
        )


class SNNBlock(nn.Module):
    """One hidden layer: Linear → BatchNorm (per time step) → Dropout → neuron."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        neuron_type: str  = "lif",
        beta: float       = 0.9,
        threshold: float  = 1.0,
        dropout: float    = 0.2,
        surrogate_slope: float = 25.0,
        time_steps: int   = 20,
    ):
        super().__init__()
        self.fc   = nn.Linear(in_features, out_features)
        self.bn   = StepBatchNorm1d(out_features, time_steps)
        self.drop = nn.Dropout(dropout)

        if _HAS_SNNTORCH:
            self.neuron = get_neuron(
                neuron_type,
                beta=beta,
                threshold=threshold,
                surrogate_slope=surrogate_slope,
            )
        else:
            self.neuron = None
            self._act   = nn.ReLU()

        self.mem: Optional[torch.Tensor] = None

    def reset_state(self):
        self.bn.reset_step()
        if _HAS_SNNTORCH and self.neuron is not None:
            self.mem = self.neuron.init_leaky()
        else:
            self.mem = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cur = self.drop(self.bn(self.fc(x)))
        if _HAS_SNNTORCH and self.neuron is not None:
            spk, self.mem = self.neuron(cur, self.mem)
        else:
            spk      = self._act(cur)
            self.mem = cur
        return spk, self.mem


class ConvSNNBlock(nn.Module):
    """Convolutional SNN block: Conv2d → BN → neuron. Used in SCNN architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int  = 3,
        padding: int      = 1,
        neuron_type: str  = "lif",
        beta: float       = 0.9,
        threshold: float  = 1.0,
        surrogate_slope: float = 25.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels)

        if _HAS_SNNTORCH:
            self.neuron = get_neuron(
                neuron_type,
                beta=beta,
                threshold=threshold,
                surrogate_slope=surrogate_slope,
            )
        else:
            self.neuron = None
            self._act   = nn.ReLU()

        self.mem: Optional[torch.Tensor] = None

    def reset_state(self):
        if _HAS_SNNTORCH and self.neuron is not None:
            self.mem = self.neuron.init_leaky()
        else:
            self.mem = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cur = self.bn(self.conv(x))
        if _HAS_SNNTORCH and self.neuron is not None:
            spk, self.mem = self.neuron(cur, self.mem)
        else:
            spk      = self._act(cur)
            self.mem = cur
        return spk, self.mem
