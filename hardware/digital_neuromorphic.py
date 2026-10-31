"""Digital Neuromorphic Computing fault model — CMOS-based SNNs."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from fault_injection.transient.bit_flip       import INT8BitFlipInjector, IntWeightFaultInjector
from fault_injection.transient.membrane_seu   import MembraneSEU, REGISTER_BITS
from fault_injection.permanent.memory_block   import MemoryBlockStuckAt
from fault_injection.framework                import FaultInjector

E_AC_PJ  = 3.5
E_MAC_PJ = 7.0

BITS             = 8
WEIGHT_NOISE_STD = 0.0


class LoihiWeightQuantization(FaultInjector):
    """Quantize Linear layer weights to 8-bit signed integers."""

    def __init__(self, bits: int = 8):
        self.bits = bits
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
            scale = w.abs().max().clamp(min=1e-8) / q_max
            w_q   = (w / scale).round().clamp(q_min, q_max) * scale
            module.weight.data = w_q

    def restore(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._saved:
                module.weight.data.copy_(self._saved[name])
        self._saved.clear()


class DigitalNeuromorphicTransientFault(INT8BitFlipInjector):
    """Transient SEU in digital neuromorphic synaptic SRAM (8-bit weights)."""

    def __init__(self, n_flips: int = 1, seed: Optional[int] = None):
        super().__init__(n_flips=n_flips, seed=seed,
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)


class DigitalNeuromorphicMembraneFault(MembraneSEU):
    """Transient SEU in the neurons' 24-bit membrane-potential registers (Loihi format), on the INT8 synaptic-memory substrate."""

    def __init__(self, n_upsets: int = 1, seed: Optional[int] = None):
        super().__init__(
            n_upsets=n_upsets, seed=seed, register_bits=REGISTER_BITS,
            substrate=IntWeightFaultInjector(n_sites=0, seed=seed, bits=BITS,
                                             weight_noise_std=WEIGHT_NOISE_STD),
        )


class DigitalNeuromorphicBlockFault(MemoryBlockStuckAt):
    """Permanent row / column block faults in the INT8 synaptic SRAM."""

    def __init__(self, n_faulty_bits: int = 1, seed: Optional[int] = None):
        super().__init__(n_faulty_bits=n_faulty_bits, seed=seed,
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)
