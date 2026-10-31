"""Probabilistic (non-deterministic) fault injection via forward hooks."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from fault_injection.framework import FaultInjector


class ProbabilisticFaultInjector(FaultInjector):
    """Hook-based probabilistic fault injection on spike/activation tensors."""

    def __init__(
        self,
        p:             float               = 1e-3,
        mode:          str                 = "zero",
        sigma:         float               = 0.1,
        target_layers: Optional[List[str]] = None,
        seed:          Optional[int]       = None,
    ):
        assert mode in ("zero", "one", "flip_sign", "noise", "bit_flip"), \
            f"Unknown mode '{mode}'"
        self.p             = p
        self.mode          = mode
        self.sigma         = sigma
        self.target_layers = target_layers
        self.seed          = seed
        self._hooks: List                   = []
        self._rng:   Optional[torch.Generator] = None

    def inject(self, model: nn.Module) -> None:
        self._hooks.clear()
        self._rng = torch.Generator()
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

        for name, module in model.named_modules():
            if self.target_layers and name not in self.target_layers:
                continue
            if type(module).__name__ != "SNNBlock":
                continue
            self._hooks.append(
                module.register_forward_hook(self._make_hook())
            )

    def restore(self, model: nn.Module) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._rng = None

    def _make_hook(self):
        def hook(module, inp, output):
            spk, mem = output if isinstance(output, tuple) else (output, None)
            spk = _corrupt(spk, self.p, self.mode, self.sigma, self._rng)
            return (spk, mem) if mem is not None else spk
        return hook


def _corrupt(
    x:     torch.Tensor,
    p:     float,
    mode:  str,
    sigma: float,
    rng:   Optional[torch.Generator],
) -> torch.Tensor:
    fault_mask = torch.bernoulli(
        torch.full(x.shape, p),
        generator=rng,
    ).bool().to(x.device)

    if not fault_mask.any():
        return x

    x = x.clone()

    if mode == "zero":
        x[fault_mask] = 0.0
    elif mode == "one":
        x[fault_mask] = 1.0
    elif mode == "flip_sign":
        x[fault_mask] = -x[fault_mask]
    elif mode == "noise":
        n_faulted = int(fault_mask.sum().item())
        noise = torch.randn(n_faulted, generator=rng).to(x.device) * sigma
        x[fault_mask] = (x[fault_mask] + noise).clamp(0.0, 1.0)
    elif mode == "bit_flip":
        flat  = x[fault_mask].float().cpu().contiguous()
        w_int = flat.view(torch.int32).clone()
        bits  = torch.randint(0, 32, (flat.numel(),), generator=rng)
        for i, b in enumerate(bits.tolist()):
            w_int[i] ^= (1 << b)
        repaired = w_int.view(torch.float32)
        repaired = torch.nan_to_num(repaired, nan=0.0, posinf=1.0, neginf=0.0)
        x[fault_mask] = repaired.to(x.device)

    return x
