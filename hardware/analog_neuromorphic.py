"""Analog Neuromorphic Computing fault model — RRAM/Memristive Synapses."""
from __future__ import annotations

from typing import Optional

from fault_injection.permanent.memristor_stuck import MemristorStuckAt

E_AC_PJ  = 0.05
E_MAC_PJ = 4.6

BITS             = 4
WEIGHT_NOISE_STD = 0.05


class AnalogNeuromorphicStuckOnFault(MemristorStuckAt):
    """Memristive synapses stuck ON (G_max)."""

    def __init__(self, n_cells: int = 1, seed: Optional[int] = None):
        super().__init__(n_cells=n_cells, seed=seed, state="on",
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)


class AnalogNeuromorphicStuckOffFault(MemristorStuckAt):
    """Memristive synapses stuck OFF (G_min)."""

    def __init__(self, n_cells: int = 1, seed: Optional[int] = None):
        super().__init__(n_cells=n_cells, seed=seed, state="off",
                         bits=BITS, weight_noise_std=WEIGHT_NOISE_STD)
