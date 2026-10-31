"""Digital In-Memory Computing (IMC) fault model."""
from __future__ import annotations

from typing import Optional

from fault_injection.transient.bit_flip       import INT8BitFlipInjector, IntWeightFaultInjector
from fault_injection.transient.membrane_seu   import MembraneSEU, REGISTER_BITS
from fault_injection.permanent.memory_block   import MemoryBlockStuckAt

E_AC_PJ  = 0.5
E_MAC_PJ = 2.0

BITS             = 4
WEIGHT_NOISE_STD = 0.0


class DigitalIMCTransientFault(INT8BitFlipInjector):
    """Transient SEU in Digital IMC weight SRAM, modelled as 4-bit integer storage."""

    def __init__(self, n_flips: int = 1, seed: Optional[int] = None):
        super().__init__(n_flips=n_flips, seed=seed,
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)


class DigitalIMCMembraneFault(MembraneSEU):
    """Transient SEU in the neurons' 24-bit membrane-potential registers, on the Digital IMC weight substrate."""

    def __init__(self, n_upsets: int = 1, seed: Optional[int] = None):
        super().__init__(
            n_upsets=n_upsets, seed=seed, register_bits=REGISTER_BITS,
            substrate=IntWeightFaultInjector(n_sites=0, seed=seed, bits=BITS,
                                             weight_noise_std=WEIGHT_NOISE_STD),
        )


class DigitalIMCBlockFault(MemoryBlockStuckAt):
    """Permanent row / column block faults in the Digital IMC weight SRAM."""

    def __init__(self, n_faulty_bits: int = 1, seed: Optional[int] = None):
        super().__init__(n_faulty_bits=n_faulty_bits, seed=seed,
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)
