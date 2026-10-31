"""Bit-flip (Single-Event Upset) fault injection."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fault_injection.framework import FaultInjector


class BitFlipInjector(FaultInjector):
    """Random bit-flip injection into Linear layer weight tensors."""

    def __init__(
        self,
        n_flips:       int                 = 1,
        bit_range:     Tuple[int, int]     = (0, 31),
        target_layers: Optional[List[str]] = None,
        seed:          Optional[int]       = None,
    ):
        self.n_flips       = n_flips
        self.bit_range     = bit_range
        self.target_layers = target_layers
        self.seed          = seed
        self._saved: Dict[str, torch.Tensor] = {}

    def inject(self, model: nn.Module) -> None:
        self._saved.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and (not self.target_layers or name in self.target_layers)
        ]
        if not layers:
            return

        n = len(layers)
        base      = self.n_flips // n
        remainder = self.n_flips % n
        flip_counts = [base + (1 if i < remainder else 0) for i in range(n)]

        for (name, module), flips_each in zip(layers, flip_counts):
            if flips_each == 0:
                continue
            w = module.weight.data
            self._saved[name] = w.clone()
            module.weight.data = _bit_flip(w, flips_each, self.bit_range, rng)

    def restore(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._saved:
                module.weight.data.copy_(self._saved[name])
        self._saved.clear()


def _distinct_flip_sites(
    n_elems: int,
    bit_lo:  int,
    bit_hi:  int,
    n_flips: int,
    rng:     torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample n_flips distinct (element, bit) sites — i.e. without replacement."""
    n_bits  = bit_hi - bit_lo + 1
    n_flips = min(n_flips, n_elems * n_bits)
    elem    = torch.randint(0, n_elems, (n_flips,), generator=rng).to(torch.int64)
    bits    = torch.randint(bit_lo, bit_hi + 1, (n_flips,), generator=rng).to(torch.int64)

    code = elem * n_bits + (bits - bit_lo)
    if torch.unique(code).numel() == code.numel():
        return elem, bits

    seen:  set       = set()
    e_out: List[int] = []
    b_out: List[int] = []
    for e, b in zip(elem.tolist(), bits.tolist()):
        while (e, b) in seen:
            e = int(torch.randint(0, n_elems, (1,), generator=rng).item())
            b = int(torch.randint(bit_lo, bit_hi + 1, (1,), generator=rng).item())
        seen.add((e, b))
        e_out.append(e)
        b_out.append(b)
    return (torch.tensor(e_out, dtype=torch.int64),
            torch.tensor(b_out, dtype=torch.int64))


def _combined_xor_mask(n: int, elem: torch.Tensor, bit: torch.Tensor) -> torch.Tensor:
    """Per-element XOR mask accumulating all flips targeting that element."""
    mask = torch.zeros(n, dtype=torch.int64)
    mask.index_add_(0, elem, torch.ones_like(bit) << bit)
    return mask


def _bit_flip(
    w:         torch.Tensor,
    n_flips:   int,
    bit_range: Tuple[int, int],
    rng:       torch.Generator,
) -> torch.Tensor:
    device   = w.device
    flat_cpu = w.detach().cpu().contiguous().flatten().clone()
    n        = flat_cpu.numel()

    elem, bit = _distinct_flip_sites(n, bit_range[0], bit_range[1], n_flips, rng)
    masks     = _combined_xor_mask(n, elem, bit)

    w_int64 = flat_cpu.view(torch.int32).to(torch.int64)
    w_int64 ^= masks
    flipped = w_int64.to(torch.int32).view(torch.float32)
    flipped = torch.nan_to_num(flipped, nan=0.0, posinf=0.0, neginf=0.0)
    return flipped.view_as(w).to(device)


INT_FAULT_TYPES = ("flip", "sa0", "sa1", "random")


def _linear_layers(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """The model's weight memories: every nn.Linear, in registration order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def _sample_memory_sites(
    sizes:   List[int],
    bits:    int,
    n_sites: int,
    rng:     torch.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Draw n_sites distinct (weight, bit) sites uniformly over the whole weight memory — all layers laid end to end — and return them split per layer."""
    empty = torch.zeros(0, dtype=torch.int64)
    if n_sites <= 0:
        return [(empty, empty) for _ in sizes]
    elem, bit = _distinct_flip_sites(sum(sizes), 0, bits - 1, n_sites, rng)
    out, offset = [], 0
    for size in sizes:
        sel = (elem >= offset) & (elem < offset + size)
        out.append((elem[sel] - offset, bit[sel]))
        offset += size
    return out


def _int_weight_fault(
    w:            torch.Tensor,
    elem:         torch.Tensor,
    bit:          torch.Tensor,
    rng:          torch.Generator,
    bits:         int   = 8,
    noise_std:    float = 0.0,
    fault_type:   str   = "flip",
    to_one:       Optional[torch.Tensor] = None,
    post_dequant: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    stats:        Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Model a `bits`-wide signed integer weight memory with optional analog device variation, and corrupt the given distinct (elem, bit) sites of it."""
    if fault_type not in INT_FAULT_TYPES:
        raise ValueError(f"fault_type must be one of {INT_FAULT_TYPES}; got '{fault_type}'")

    device = w.device
    w_cpu  = w.detach().cpu().contiguous()

    q_max  = (1 << (bits - 1)) - 1
    q_min  = -(1 << (bits - 1))
    absmax = w_cpu.abs().max().clamp(min=1e-8)
    scale  = absmax / q_max

    q    = (w_cpu / scale).round().clamp(q_min, q_max).to(torch.int64)
    flat = q.flatten().clone()
    n    = flat.numel()

    if elem.numel() > 0:
        mask      = (1 << bits) - 1
        sign_bit  = 1 << (bits - 1)

        v = flat & mask
        before = v.clone() if stats is not None else None
        if fault_type == "flip":
            v ^= _combined_xor_mask(n, elem, bit)
        else:
            if to_one is None:
                if fault_type == "random":
                    to_one = torch.randint(0, 2, (elem.numel(),), generator=rng).bool()
                else:
                    to_one = torch.full((elem.numel(),), fault_type == "sa1")
            set_mask   = _combined_xor_mask(n, elem[to_one],  bit[to_one])
            clear_mask = _combined_xor_mask(n, elem[~to_one], bit[~to_one])
            v = (v & ~clear_mask) | set_mask
        v  &= mask
        if stats is not None:
            touched = torch.unique(elem)
            diff = before[touched] ^ v[touched]
            changed = sum(int(((diff >> b) & 1).sum()) for b in range(bits))
            stats["placed"] = stats.get("placed", 0) + int(elem.numel())
            stats["changed"] = stats.get("changed", 0) + changed
        v   = torch.where((v & sign_bit) != 0, v - (1 << bits), v)
        flat = v

    deq = flat.to(torch.float32).view_as(w_cpu) * scale

    if post_dequant is not None:
        deq = post_dequant(deq)

    if noise_std > 0.0:
        noise = torch.randn(deq.shape, generator=rng) * (noise_std * absmax)
        deq   = deq + noise

    return deq.to(device)


class IntWeightFaultInjector(FaultInjector):
    """Low-precision integer weight memory with n_sites corrupted bits."""

    def __init__(
        self,
        n_sites:          int             = 1,
        seed:             Optional[int]   = None,
        bits:             int             = 8,
        weight_noise_std: float           = 0.0,
        fault_type:       str             = "flip",
    ):
        if fault_type not in INT_FAULT_TYPES:
            raise ValueError(f"fault_type must be one of {INT_FAULT_TYPES}; got '{fault_type}'")
        self.n_sites          = n_sites
        self.seed             = seed
        self.bits             = bits
        self.weight_noise_std = weight_noise_std
        self.fault_type       = fault_type
        self._saved: Dict[str, torch.Tensor] = {}
        self._trials = 0
        self._counts: Dict[str, int] = {}

    def fault_stats(self) -> Dict[str, float]:
        """Per-trial means over every trial run with this injector: "placed" fault sites and "changed" stored bits."""
        trials = max(self._trials, 1)
        return {k: v / trials for k, v in self._counts.items()}

    def fault_sites(self, model: nn.Module, time_steps: Optional[int] = None) -> int:
        """Denominator of the fault rate: every bit stored in the weight memory."""
        return sum(m.weight.numel() for _, m in _linear_layers(model)) * self.bits

    def _plan(self, layers: List[Tuple[str, nn.Linear]],
              rng: torch.Generator) -> List[Dict[str, Any]]:
        """Per-layer fault plan: {"elem", "bit"} and optionally "to_one" (stuck values) and "post" (a value-level fault)."""
        sites = _sample_memory_sites(
            [m.weight.numel() for _, m in layers], self.bits, self.n_sites, rng
        )
        return [{"elem": elem, "bit": bit} for elem, bit in sites]

    def _bit_fault(self) -> str:
        """What happens at a bit-level site; subclasses with no bit sites override."""
        return self.fault_type

    def inject(self, model: nn.Module) -> None:
        self._saved.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        layers = _linear_layers(model)
        if not layers:
            return
        self._trials += 1

        plan = self._plan(layers, rng)

        for (name, module), p in zip(layers, plan):
            w = module.weight.data
            self._saved[name] = w.clone()
            module.weight.data = _int_weight_fault(
                w, p["elem"], p["bit"], rng,
                bits=self.bits, noise_std=self.weight_noise_std,
                fault_type=self._bit_fault(),
                to_one=p.get("to_one"), post_dequant=p.get("post"),
                stats=self._counts,
            )

    def restore(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._saved:
                module.weight.data.copy_(self._saved[name])
        self._saved.clear()


class INT8BitFlipInjector(IntWeightFaultInjector):
    """Transient SEU: n_flips bits of the integer weight memory are toggled."""

    def __init__(
        self,
        n_flips:          int             = 1,
        seed:             Optional[int]   = None,
        bits:             int             = 8,
        weight_noise_std: float           = 0.0,
    ):
        super().__init__(n_sites=n_flips, seed=seed, bits=bits,
                         weight_noise_std=weight_noise_std, fault_type="flip")
