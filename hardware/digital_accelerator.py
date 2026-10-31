"""CMOS / FPGA / systolic-array digital accelerator fault models."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fault_injection.framework                  import FaultInjector
from fault_injection.permanent.stuck_at_weights import StuckAtWeights


class AcceleratorQuantization(FaultInjector):
    """Uniform affine quantization of Linear layer weights to N bits."""

    def __init__(self, bits: int = 8, per_channel: bool = False):
        assert bits in (4, 8, 16), f"bits must be 4, 8, or 16; got {bits}"
        self.bits        = bits
        self.per_channel = per_channel
        self._saved: Dict[str, torch.Tensor] = {}

    def inject(self, model: nn.Module) -> None:
        self._saved.clear()
        q_min = -(2 ** (self.bits - 1))
        q_max =  (2 ** (self.bits - 1)) - 1

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.data
            self._saved[name] = w.clone()

            if self.per_channel:
                scale = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / q_max
            else:
                scale = w.abs().max().clamp(min=1e-8) / q_max

            w_q = (w / scale).round().clamp(q_min, q_max) * scale
            module.weight.data = w_q

    def restore(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._saved:
                module.weight.data.copy_(self._saved[name])
        self._saved.clear()


class MACStuckAt(FaultInjector):
    """Stuck-at faults in the output of MAC (multiply-accumulate) units."""

    def __init__(
        self,
        fault_rate:    float               = 1e-4,
        fault_type:    str                 = "sa0",
        sat_value:     float               = 1.0,
        target_layers: Optional[List[str]] = None,
        seed:          Optional[int]       = None,
    ):
        assert fault_type in ("sa0", "sa1", "random")
        self.fault_rate    = fault_rate
        self.fault_type    = fault_type
        self.sat_value     = sat_value
        self.target_layers = target_layers
        self.seed          = seed
        self._hooks: List                     = []
        self._masks: Dict[str, torch.Tensor]  = {}

    def inject(self, model: nn.Module) -> None:
        self._hooks.clear()
        self._masks.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.target_layers and name not in self.target_layers:
                continue

            n_out    = module.out_features
            n_faulty = int(n_out * self.fault_rate)
            if n_faulty == 0:
                continue

            perm     = torch.randperm(n_out, generator=rng)[:n_faulty]
            mask     = torch.zeros(n_out, dtype=torch.bool)
            mask[perm] = True
            self._masks[name] = mask

            self._hooks.append(
                module.register_forward_hook(
                    _make_mac_hook(mask, self.fault_type, self.sat_value)
                )
            )

    def restore(self, model: nn.Module) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._masks.clear()


def _make_mac_hook(mask: torch.Tensor, fault_type: str, sat_value: float):
    def hook(module, inp, output):
        out = output.clone()
        m   = mask.to(out.device)
        if fault_type == "sa0":
            out[..., m] = 0.0
        elif fault_type == "sa1":
            out[..., m] = sat_value
        else:
            idxs = m.nonzero(as_tuple=False).squeeze(1)
            half = len(idxs) // 2
            out[..., idxs[:half]] = 0.0
            out[..., idxs[half:]] = sat_value
        return out
    return hook


class WeightSRAMError(StuckAtWeights):
    """Bit errors in weight SRAM cells (wear-out or radiation)."""

    def __init__(
        self,
        fault_rate: float         = 1e-6,
        fault_type: str           = "random",
        seed:       Optional[int] = None,
    ):
        super().__init__(
            fault_rate=fault_rate,
            fault_type=fault_type,
            bit_range=(0, 30),
            seed=seed,
        )


class DigitalAcceleratorFault(FaultInjector):
    """Combined fault profile for a typical INT8 digital inference accelerator."""

    def __init__(
        self,
        bits:       Optional[int]   = 8,
        mac_rate:   Optional[float] = 1e-4,
        sram_rate:  Optional[float] = 1e-6,
        fault_type: str             = "random",
        seed:       Optional[int]   = None,
    ):
        self._injectors: List[FaultInjector] = []
        if bits is not None:
            self._injectors.append(AcceleratorQuantization(bits=bits))
        if mac_rate is not None:
            self._injectors.append(
                MACStuckAt(fault_rate=mac_rate, fault_type=fault_type, seed=seed)
            )
        if sram_rate is not None:
            self._injectors.append(
                WeightSRAMError(fault_rate=sram_rate, fault_type=fault_type, seed=seed)
            )

    def inject(self, model: nn.Module) -> None:
        for inj in self._injectors:
            inj.inject(model)

    def restore(self, model: nn.Module) -> None:
        for inj in reversed(self._injectors):
            inj.restore(model)
