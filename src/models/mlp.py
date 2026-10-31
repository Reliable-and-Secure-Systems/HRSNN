from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from src.models.base   import BaseSNN
from src.models.blocks import SNNBlock


class SNNMLP(BaseSNN):
    """SNN MLP with configurable depth, width, and neuron type."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        n_classes: int         = 10,
        neuron_type: str       = "lif",
        beta: float            = 0.9,
        threshold: float       = 1.0,
        dropout: float         = 0.2,
        surrogate_slope: float = 25.0,
        time_steps: int        = 20,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128, 64]
        dims        = [input_dim] + hidden_dims

        self.blocks = nn.ModuleList([
            SNNBlock(
                dims[i], dims[i + 1],
                neuron_type=neuron_type,
                beta=beta,
                threshold=threshold,
                dropout=dropout,
                surrogate_slope=surrogate_slope,
                time_steps=time_steps,
            )
            for i in range(len(dims) - 1)
        ])
        self.output_fc = nn.Linear(dims[-1], n_classes)
        self.n_classes = n_classes

    def reset_states(self):
        for block in self.blocks:
            block.reset_state()

    def forward(self, x: torch.Tensor):
        """Run the MLP over T steps; returns the readout logits per step [T, B, n_classes] twice."""
        self.reset_states()
        logit_rec = []
        for t in range(x.shape[0]):
            spk = x[t]
            for block in self.blocks:
                spk, _ = block(spk)
            logit_rec.append(self.output_fc(spk))
        stacked = torch.stack(logit_rec)
        return stacked, stacked

    def architecture_summary(self) -> str:
        lines = ["─" * 50, f"  SNNMLP  blocks={len(self.blocks)}", "─" * 50]
        for i, b in enumerate(self.blocks):
            lines.append(
                f"  Layer {i+1}: {b.fc.in_features} → {b.fc.out_features}"
            )
        lines += [
            f"  Output: {self.output_fc.in_features} → {self.output_fc.out_features}",
            f"  Params: {self.count_parameters():,}",
            "─" * 50,
        ]
        return "\n".join(lines)
