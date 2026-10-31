from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseSNN(nn.Module, ABC):
    """Contract for all SNN models."""

    @abstractmethod
    def reset_states(self):
        """Reset all stateful neuron membrane potentials."""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Run the network over T steps; returns (output, aux), each [T, B, n_classes]."""

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
