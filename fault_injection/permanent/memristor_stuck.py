"""Permanent stuck memristor cells in an analog crossbar."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fault_injection.transient.bit_flip import (
    IntWeightFaultInjector, _linear_layers, _sample_memory_sites,
)

STATES = ("on", "off")


def stuck_cells(w: torch.Tensor, elem: torch.Tensor, cell: torch.Tensor,
                state: str, stats: Optional[Dict[str, int]] = None) -> torch.Tensor:
    """Apply stuck cells to dequantized weights w (any shape)."""
    flat = w.flatten().clone()
    full_scale = flat.abs().max().clamp(min=1e-8)
    g_pos = flat.clamp(min=0) / full_scale
    g_neg = (-flat).clamp(min=0) / full_scale
    level = 1.0 if state == "on" else 0.0
    if stats is not None:
        changed = int((g_pos[elem[cell == 0]] != level).sum()) + \
                  int((g_neg[elem[cell == 1]] != level).sum())
        stats["placed"] = stats.get("placed", 0) + int(elem.numel())
        stats["changed"] = stats.get("changed", 0) + changed
    g_pos[elem[cell == 0]] = level
    g_neg[elem[cell == 1]] = level
    return ((g_pos - g_neg) * full_scale).view_as(w)


class MemristorStuckAt(IntWeightFaultInjector):
    """Memristor cells stuck ON (G_max) or OFF (G_min) in a differential-pair crossbar."""

    def __init__(
        self,
        n_cells:          int           = 1,
        seed:             Optional[int] = None,
        state:            str           = "on",
        bits:             int           = 4,
        weight_noise_std: float         = 0.0,
    ):
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}; got '{state}'")
        super().__init__(n_sites=n_cells, seed=seed, bits=bits,
                         weight_noise_std=weight_noise_std, fault_type="flip")
        self.state = state
        self.fault_type = f"stuck_{state}"

    def _bit_fault(self) -> str:
        return "flip"

    def fault_sites(self, model: nn.Module, time_steps: Optional[int] = None) -> int:
        """Denominator of the rate: two memristor cells per weight."""
        return 2 * sum(m.weight.numel() for _, m in _linear_layers(model))

    def _plan(self, layers: List[Tuple[str, nn.Linear]],
              rng: torch.Generator) -> List[Dict[str, Any]]:
        sites = _sample_memory_sites(
            [m.weight.numel() for _, m in layers], 2, self.n_sites, rng
        )
        empty = torch.zeros(0, dtype=torch.int64)
        plan = []
        for elem, cell in sites:
            post: Optional[Callable] = None
            if elem.numel():
                post = (lambda w, e=elem, c=cell:
                        stuck_cells(w, e, c, self.state, stats=self._counts))
            plan.append({"elem": empty, "bit": empty, "post": post})
        return plan
