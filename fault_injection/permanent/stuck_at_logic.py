"""Stuck-at faults in compute fabric (neuron logic / spike wires)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fault_injection.framework import FaultInjector


class StuckAtLogic(FaultInjector):
    """SA0/SA1 stuck-at faults on spike outputs of SNNBlock layers."""

    def __init__(
        self,
        fault_rate:    float               = 1e-3,
        fault_type:    str                 = "sa0",
        target_layers: Optional[List[str]] = None,
        seed:          Optional[int]       = None,
    ):
        assert fault_type in ("sa0", "sa1", "random"), \
            f"fault_type must be 'sa0', 'sa1', or 'random'; got '{fault_type}'"
        self.fault_rate    = fault_rate
        self.fault_type    = fault_type
        self.target_layers = target_layers
        self.seed          = seed
        self._hooks: List  = []
        self._masks: Dict[str, torch.Tensor] = {}

    def inject(self, model: nn.Module) -> None:
        self._hooks.clear()
        self._masks.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        for name, module in model.named_modules():
            if self.target_layers and name not in self.target_layers:
                continue
            if type(module).__name__ != "SNNBlock":
                continue

            fc = getattr(module, "fc", None)
            if not isinstance(fc, nn.Linear):
                continue

            n_neurons = fc.out_features
            mask      = _make_mask(n_neurons, self.fault_rate, rng)
            self._masks[name] = mask
            self._hooks.append(
                module.register_forward_hook(
                    _make_hook(mask, self.fault_type)
                )
            )

    def restore(self, model: nn.Module) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._masks.clear()


def _make_mask(n: int, fault_rate: float, rng: torch.Generator) -> torch.Tensor:
    n_faulty = max(1, int(n * fault_rate))
    perm     = torch.randperm(n, generator=rng)[:n_faulty]
    mask     = torch.zeros(n, dtype=torch.bool)
    mask[perm] = True
    return mask


def _make_hook(mask: torch.Tensor, fault_type: str):
    def hook(module, inp, output):
        spk, mem = output if isinstance(output, tuple) else (output, None)
        m   = mask.to(spk.device)
        spk = spk.clone()

        if fault_type == "sa0":
            spk[..., m] = 0.0
        elif fault_type == "sa1":
            spk[..., m] = 1.0
        else:
            idxs = m.nonzero(as_tuple=False).squeeze(1)
            half = len(idxs) // 2
            spk[..., idxs[:half]] = 0.0
            spk[..., idxs[half:]] = 1.0

        return (spk, mem) if mem is not None else spk
    return hook
