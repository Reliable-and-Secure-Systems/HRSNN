"""Transient upsets in the neuron state: bit flips in membrane-potential registers."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fault_injection.framework import FaultInjector
from fault_injection.transient.bit_flip import _distinct_flip_sites

REGISTER_BITS  = 24
REGISTER_RANGE = 256


def _snn_blocks(model: nn.Module) -> List[nn.Module]:
    """Spiking layers that hold a membrane register, in forward order."""
    return [m for m in model.modules()
            if type(m).__name__ == "SNNBlock" and getattr(m, "neuron", None) is not None]


def register_lsb(threshold: float, register_bits: int = REGISTER_BITS,
                 register_range: float = REGISTER_RANGE) -> float:
    """Value of one register LSB when the register spans ±register_range·θ."""
    return register_range * threshold / (1 << (register_bits - 1))


def flip_membrane_bits(
    mem:       torch.Tensor,
    neurons:   torch.Tensor,
    xor_masks: torch.Tensor,
    lsb:       float,
    reg_bits:  int,
) -> torch.Tensor:
    """Flip bits of the fixed-point membrane register of the given neurons."""
    q_max = (1 << (reg_bits - 1)) - 1
    q_min = -(1 << (reg_bits - 1))
    mask = (1 << reg_bits) - 1
    sign = 1 << (reg_bits - 1)

    out = mem.clone()
    v = out[:, neurons].double()
    q = torch.round(v / lsb).clamp(q_min, q_max).to(torch.int64)
    u = (q & mask) ^ xor_masks.to(q.device).unsqueeze(0)
    u = torch.where((u & sign) != 0, u - (1 << reg_bits), u)
    out[:, neurons] = (u.double() * lsb).to(mem.dtype)
    return out


class MembraneSEU(FaultInjector):
    """Bit flips in neuron membrane-potential registers at random time steps."""

    def __init__(
        self,
        n_upsets:       int                     = 1,
        seed:           Optional[int]           = None,
        register_bits:  int                     = REGISTER_BITS,
        register_range: float                   = REGISTER_RANGE,
        substrate:      Optional[FaultInjector] = None,
    ):
        self.n_upsets       = n_upsets
        self.seed           = seed
        self.register_bits  = register_bits
        self.register_range = register_range
        self.substrate      = substrate
        self.fault_type     = "membrane_flip"
        self._hooks: List = []
        self._rng: Optional[torch.Generator] = None
        self._steps: Dict[int, int] = {}
        self._upsets: Optional[Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]] = None
        self._trials = 0
        self._placed = 0
        self._hits = 0
        self._clamped = 0

    @property
    def bits(self) -> Optional[int]:
        return getattr(self.substrate, "bits", None)

    @property
    def weight_noise_std(self) -> float:
        return getattr(self.substrate, "weight_noise_std", 0.0)

    def fault_sites(self, model: nn.Module, time_steps: Optional[int] = None) -> int:
        """Denominator of the rate: neurons x register bits x (time steps - 1)."""
        if time_steps is None:
            raise ValueError("MembraneSEU.fault_sites needs time_steps")
        neurons = sum(b.fc.out_features for b in _snn_blocks(model))
        return neurons * self.register_bits * max(time_steps - 1, 1)

    def fault_stats(self) -> Dict[str, float]:
        """Per-trial means of the upsets placed and of the share of clamped membrane values."""
        trials = max(self._trials, 1)
        return {
            "placed":           self._placed / trials,
            "clamped_fraction": self._clamped / max(self._hits, 1),
        }

    def inject(self, model: nn.Module) -> None:
        self._hooks.clear()
        self._rng = torch.Generator()
        if self.seed is not None:
            self._rng.manual_seed(self.seed)
        self._upsets = None
        self._trials += 1

        if self.substrate is not None:
            self.substrate.seed = self.seed
            self.substrate.inject(model)

        blocks = _snn_blocks(model)
        sizes = [b.fc.out_features for b in blocks]

        def start_inference(module, args):
            self._steps = {k: 0 for k in range(len(blocks))}
            if self._upsets is None:
                self._upsets = self._draw(sizes, args[0].shape[0])

        self._hooks.append(model.register_forward_pre_hook(start_inference))
        for k, block in enumerate(blocks):
            self._hooks.append(block.register_forward_hook(self._make_hook(k)))

    def _draw(self, sizes: List[int], time_steps: int):
        """n upsets, uniform and distinct over (neuron, bit, step 0 … T-2)."""
        upsets: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        n_neurons = sum(sizes)
        window = max(time_steps - 1, 1)
        if self.n_upsets <= 0 or n_neurons == 0:
            return upsets
        elem, bit = _distinct_flip_sites(n_neurons * window, 0,
                                         self.register_bits - 1, self.n_upsets, self._rng)
        self._placed += elem.numel()
        t_all, neuron_all = elem // n_neurons, elem % n_neurons
        offsets = [sum(sizes[:k]) for k in range(len(sizes))]
        for k, (off, size) in enumerate(zip(offsets, sizes)):
            in_block = (neuron_all >= off) & (neuron_all < off + size)
            for t in torch.unique(t_all[in_block]).tolist():
                sel = in_block & (t_all == t)
                local = neuron_all[sel] - off
                masks = torch.ones_like(bit[sel]) << bit[sel]
                uniq, inv = torch.unique(local, return_inverse=True)
                combined = torch.zeros(uniq.numel(), dtype=torch.int64)
                combined.index_add_(0, inv, masks)
                upsets.setdefault(k, {})[t] = (uniq, combined)
        return upsets

    def _make_hook(self, k: int):
        def hook(module, inp, output):
            step = self._steps[k]
            self._steps[k] = step + 1
            hits = (self._upsets or {}).get(k, {}).get(step)
            if hits is None:
                return None
            neurons, masks = hits
            neurons = neurons.to(module.mem.device)
            theta = float(module.neuron.threshold)
            lsb = register_lsb(theta, self.register_bits, self.register_range)
            limit = ((1 << (self.register_bits - 1)) - 1) * lsb
            values = module.mem[:, neurons]
            self._hits += values.numel()
            self._clamped += int((values.abs() > limit).sum().item())
            module.mem = flip_membrane_bits(module.mem, neurons, masks, lsb,
                                            self.register_bits)
            return None
        return hook

    def restore(self, model: nn.Module) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._upsets = None
        if self.substrate is not None:
            self.substrate.restore(model)
