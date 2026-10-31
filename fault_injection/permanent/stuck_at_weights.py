"""Stuck-at faults in synaptic weight memory (SRAM bit cells)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fault_injection.framework          import FaultInjector
from fault_injection.transient.bit_flip import IntWeightFaultInjector


class IntStuckAtWeights(IntWeightFaultInjector):
    """Permanent stuck-at bits in low-precision integer weight memory."""

    def __init__(
        self,
        n_faults:         int           = 1,
        seed:             Optional[int] = None,
        bits:             int           = 8,
        weight_noise_std: float         = 0.0,
        fault_type:       str           = "random",
    ):
        if fault_type not in ("sa0", "sa1", "random"):
            raise ValueError(
                f"fault_type must be 'sa0', 'sa1' or 'random'; got '{fault_type}'"
            )
        super().__init__(n_sites=n_faults, seed=seed, bits=bits,
                         weight_noise_std=weight_noise_std, fault_type=fault_type)


class StuckAtWeights(FaultInjector):
    """SA0/SA1 stuck-at faults in Linear layer weight tensors."""

    def __init__(
        self,
        fault_rate:    float               = 1e-4,
        fault_type:    str                 = "random",
        bit_range:     Tuple[int, int]     = (0, 30),
        target_layers: Optional[List[str]] = None,
        seed:          Optional[int]       = None,
    ):
        assert fault_type in ("sa0", "sa1", "random"), \
            f"fault_type must be 'sa0', 'sa1', or 'random'; got '{fault_type}'"
        self.fault_rate    = fault_rate
        self.fault_type    = fault_type
        self.bit_range     = bit_range
        self.target_layers = target_layers
        self.seed          = seed
        self._saved: Dict[str, torch.Tensor] = {}

    def inject(self, model: nn.Module) -> None:
        self._saved.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.target_layers and name not in self.target_layers:
                continue

            w = module.weight.data
            self._saved[name] = w.clone()
            module.weight.data = _apply_stuck_at(
                w, self.fault_rate, self.fault_type, self.bit_range, rng
            )

    def restore(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._saved:
                module.weight.data.copy_(self._saved[name])
        self._saved.clear()


def _apply_stuck_at(
    w:          torch.Tensor,
    fault_rate: float,
    fault_type: str,
    bit_range:  Tuple[int, int],
    rng:        torch.Generator,
) -> torch.Tensor:
    device   = w.device
    flat_cpu = w.detach().cpu().contiguous().flatten().clone()
    n_elem   = flat_cpu.numel()
    n_bits   = n_elem * 32
    n_faults = max(1, int(n_bits * fault_rate))

    elem_idx = torch.randint(0, n_elem, (n_faults,), generator=rng)
    bit_pos  = torch.randint(
        bit_range[0], bit_range[1] + 1, (n_faults,), generator=rng
    )

    w_int = flat_cpu.view(torch.int32).clone()

    for idx, bit in zip(elem_idx.tolist(), bit_pos.tolist()):
        mask = 1 << bit
        if fault_type == "sa0":
            w_int[idx] &= ~mask
        elif fault_type == "sa1":
            w_int[idx] |= mask
        else:
            if torch.randint(0, 2, (1,), generator=rng).item():
                w_int[idx] |= mask
            else:
                w_int[idx] &= ~mask

    corrupted = w_int.view(torch.float32)
    corrupted = torch.nan_to_num(corrupted, nan=0.0, posinf=0.0, neginf=0.0)
    return corrupted.view_as(w).to(device)
