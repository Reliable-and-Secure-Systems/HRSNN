"""Permanent faults in memory blocks: whole rows or columns of the weight memory."""
from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fault_injection.transient.bit_flip import IntWeightFaultInjector

TILE = 256
ROW_SHARE = 8.2 / (8.2 + 5.6)


class MemoryBlockStuckAt(IntWeightFaultInjector):
    """Permanent row / column block faults in integer weight memory."""

    def __init__(
        self,
        n_faulty_bits:    int           = 1,
        seed:             Optional[int] = None,
        bits:             int           = 8,
        weight_noise_std: float         = 0.0,
        tile:             int           = TILE,
        row_share:        float         = ROW_SHARE,
    ):
        super().__init__(n_sites=n_faulty_bits, seed=seed, bits=bits,
                         weight_noise_std=weight_noise_std, fault_type="random")
        self.tile       = tile
        self.row_share  = row_share
        self.fault_type = "block_stuck"
        self.blocks: List[Tuple[str, int, int]] = []

    def _bit_fault(self) -> str:
        return "random"


    def _row_count(self, out_f: int, in_f: int) -> int:
        return in_f * -(-out_f // self.tile)

    def _col_count(self, out_f: int, in_f: int) -> int:
        return out_f * self.bits * -(-in_f // self.tile)

    def _row_cells(self, out_f: int, in_f: int, idx: int):
        """Row block idx -> (elem, bit): one input, one tile of outputs, all bits."""
        n_tiles = -(-out_f // self.tile)
        i, t = divmod(idx, n_tiles)
        outs = torch.arange(t * self.tile, min((t + 1) * self.tile, out_f))
        elem = (outs * in_f + i).repeat_interleave(self.bits)
        bit = torch.arange(self.bits).repeat(outs.numel())
        return elem, bit

    def _col_cells(self, out_f: int, in_f: int, idx: int):
        """Column block idx -> (elem, bit): one output, one bit, one tile of inputs."""
        n_tiles = -(-in_f // self.tile)
        o_bit, t = divmod(idx, n_tiles)
        o, b = divmod(o_bit, self.bits)
        ins = torch.arange(t * self.tile, min((t + 1) * self.tile, in_f))
        elem = o * in_f + ins
        bit = torch.full((ins.numel(),), b, dtype=torch.int64)
        return elem, bit


    def _plan(self, layers: List[Tuple[str, nn.Linear]],
              rng: torch.Generator) -> List[Dict[str, Any]]:
        shapes = [tuple(m.weight.shape) for _, m in layers]
        rows = [self._row_count(o, i) for o, i in shapes]
        cols = [self._col_count(o, i) for o, i in shapes]
        row_cum = [sum(rows[:k + 1]) for k in range(len(rows))]
        col_cum = [sum(cols[:k + 1]) for k in range(len(cols))]
        total_rows, total_cols = row_cum[-1], col_cum[-1]

        per_layer: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = \
            [[] for _ in layers]
        used: set = set()
        self.blocks = []
        faulty = 0
        target = self.n_sites

        while faulty < target and len(used) < total_rows + total_cols:
            is_row = torch.rand(1, generator=rng).item() < self.row_share
            kind, total, cum = ("row", total_rows, row_cum) if is_row \
                else ("col", total_cols, col_cum)
            g = int(torch.randint(0, total, (1,), generator=rng).item())
            if (kind, g) in used:
                continue
            used.add((kind, g))

            layer = bisect.bisect_right(cum, g)
            local = g - (cum[layer - 1] if layer else 0)
            out_f, in_f = shapes[layer]
            elem, bit = (self._row_cells if is_row else self._col_cells)(out_f, in_f, local)
            value = bool(torch.randint(0, 2, (1,), generator=rng).item())
            per_layer[layer].append((elem, bit, torch.full((elem.numel(),), value)))
            self.blocks.append((kind, layer, local))
            faulty += elem.numel()

        plan = []
        empty = torch.zeros(0, dtype=torch.int64)
        for k, chunks in enumerate(per_layer):
            if not chunks:
                plan.append({"elem": empty, "bit": empty})
                continue
            elem = torch.cat([c[0] for c in chunks])
            bit = torch.cat([c[1] for c in chunks])
            to_one = torch.cat([c[2] for c in chunks])
            code = elem * self.bits + bit
            order = torch.arange(code.numel())
            _, inverse = torch.unique(code, return_inverse=True)
            last = torch.full((int(inverse.max()) + 1,), -1, dtype=torch.int64)
            last.scatter_reduce_(0, inverse, order, reduce="amax")
            keep = last[inverse] == order
            plan.append({"elem": elem[keep], "bit": bit[keep], "to_one": to_one[keep]})
        return plan
