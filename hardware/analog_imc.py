"""Analog In-Memory Computing (IMC) fault model — RRAM Crossbar Arrays."""
from __future__ import annotations

from typing import Optional

from fault_injection.permanent.memristor_stuck import MemristorStuckAt

E_AC_PJ  = 0.05
E_MAC_PJ = 4.6

BITS             = 6
WEIGHT_NOISE_STD = 0.02


class AnalogIMCStuckOnFault(MemristorStuckAt):
    """RRAM cells stuck ON (G_max) in the Analog IMC crossbar."""

    def __init__(self, n_cells: int = 1, seed: Optional[int] = None):
        super().__init__(n_cells=n_cells, seed=seed, state="on",
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)


class AnalogIMCStuckOffFault(MemristorStuckAt):
    """RRAM cells stuck OFF (G_min) in the Analog IMC crossbar."""

    def __init__(self, n_cells: int = 1, seed: Optional[int] = None):
        super().__init__(n_cells=n_cells, seed=seed, state="off",
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)
