"""Analog and mixed-signal hardware fault models."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fault_injection.framework import FaultInjector


class DeviceMismatch(FaultInjector):
    """Additive Gaussian weight perturbation modelling device mismatch."""

    def __init__(
        self,
        sigma_rel: float         = 0.02,
        relative:  bool          = True,
        seed:      Optional[int] = None,
    ):
        self.sigma_rel = sigma_rel
        self.relative  = relative
        self.seed      = seed
        self._saved: Dict[str, torch.Tensor] = {}

    def inject(self, model: nn.Module) -> None:
        self._saved.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.data
            self._saved[name] = w.clone()

            if self.relative:
                sigma = self.sigma_rel * w.std().item()
            else:
                sigma = self.sigma_rel

            noise = torch.randn(w.shape, generator=rng).to(w.device) * sigma
            module.weight.data = w + noise

    def restore(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._saved:
                module.weight.data.copy_(self._saved[name])
        self._saved.clear()


class ProcessVariation(FaultInjector):
    """Per-neuron membrane decay (β) variation from transistor Vth spread."""

    def __init__(
        self,
        sigma_beta: float         = 0.02,
        beta_min:   float         = 0.5,
        beta_max:   float         = 0.999,
        seed:       Optional[int] = None,
    ):
        self.sigma_beta = sigma_beta
        self.beta_min   = beta_min
        self.beta_max   = beta_max
        self.seed       = seed
        self._saved_betas: Dict[str, object] = {}

    def inject(self, model: nn.Module) -> None:
        import snntorch as snn
        self._saved_betas.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        for name, module in model.named_modules():
            if not isinstance(module, snn.Leaky):
                continue
            beta_orig = module.beta
            self._saved_betas[name] = beta_orig

            beta_val = float(beta_orig) if not isinstance(beta_orig, torch.Tensor) \
                       else beta_orig.item()
            delta    = torch.randn(1, generator=rng).item() * self.sigma_beta
            beta_new = float(max(self.beta_min, min(self.beta_max, beta_val + delta)))
            module.beta = beta_new

    def restore(self, model: nn.Module) -> None:
        import snntorch as snn
        for name, module in model.named_modules():
            if isinstance(module, snn.Leaky) and name in self._saved_betas:
                module.beta = self._saved_betas[name]
        self._saved_betas.clear()


class ThermalNoise(FaultInjector):
    """Additive Gaussian noise on neuron activations, modelling kT/C thermal noise."""

    def __init__(self, sigma: float = 0.01, seed: Optional[int] = None):
        self.sigma = sigma
        self.seed  = seed
        self._hooks: List                      = []
        self._rng:   Optional[torch.Generator] = None

    def inject(self, model: nn.Module) -> None:
        import snntorch as snn
        self._hooks.clear()
        self._rng = torch.Generator()
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

        for _, module in model.named_modules():
            if not isinstance(module, snn.Leaky):
                continue
            self._hooks.append(
                module.register_forward_pre_hook(self._make_hook())
            )

    def restore(self, model: nn.Module) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._rng = None

    def _make_hook(self):
        def pre_hook(module, args):
            x     = args[0]
            noise = torch.randn(x.shape, generator=self._rng).to(x.device) * self.sigma
            return (x + noise,) + tuple(args[1:])
        return pre_hook


class WriteDrift(FaultInjector):
    """Multiplicative conductance drift after programming in memristive devices."""

    def __init__(self, drift_rate: float = 0.05, seed: Optional[int] = None):
        self.drift_rate = drift_rate
        self.seed       = seed
        self._saved: Dict[str, torch.Tensor] = {}

    def inject(self, model: nn.Module) -> None:
        self._saved.clear()
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            w = module.weight.data
            self._saved[name] = w.clone()
            drift  = 1.0 + torch.randn(w.shape, generator=rng).to(w.device) * self.drift_rate
            module.weight.data = w * drift

    def restore(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self._saved:
                module.weight.data.copy_(self._saved[name])
        self._saved.clear()


class AnalogFaultModel(FaultInjector):
    """Combined analog/mixed-signal hardware fault profile."""

    def __init__(
        self,
        sigma_mismatch: Optional[float] = 0.02,
        sigma_beta:     Optional[float] = 0.02,
        sigma_thermal:  Optional[float] = 0.01,
        drift_rate:     Optional[float] = None,
        seed:           Optional[int]   = None,
    ):
        self._injectors: List[FaultInjector] = []
        if sigma_mismatch is not None:
            self._injectors.append(
                DeviceMismatch(sigma_rel=sigma_mismatch, seed=seed)
            )
        if sigma_beta is not None:
            self._injectors.append(
                ProcessVariation(sigma_beta=sigma_beta, seed=seed)
            )
        if sigma_thermal is not None:
            self._injectors.append(ThermalNoise(sigma=sigma_thermal, seed=seed))
        if drift_rate is not None:
            self._injectors.append(WriteDrift(drift_rate=drift_rate, seed=seed))

    def inject(self, model: nn.Module) -> None:
        for inj in self._injectors:
            inj.inject(model)

    def restore(self, model: nn.Module) -> None:
        for inj in reversed(self._injectors):
            inj.restore(model)
