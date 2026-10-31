"""Fault injection framework — base class and evaluation utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FaultInjector(ABC):
    """Abstract base for all fault injection strategies."""

    @abstractmethod
    def inject(self, model: nn.Module) -> None:
        """Apply fault to model in-place. Must be paired with restore()."""

    @abstractmethod
    def restore(self, model: nn.Module) -> None:
        """Remove all fault effects applied by inject()."""

    def __call__(self, model: nn.Module) -> "FaultInjector":
        self._ctx_model = model
        return self

    def __enter__(self) -> "FaultInjector":
        self.inject(self._ctx_model)
        return self

    def __exit__(self, *args) -> None:
        self.restore(self._ctx_model)


def evaluate_under_fault(
    model:        nn.Module,
    loader:       DataLoader,
    device:       torch.device,
    injector:     FaultInjector,
    n_trials:     int            = 10,
    baseline_acc: Optional[float] = None,
    seed:         Optional[int]   = None,
    max_batches:  Optional[int]   = None,
) -> Dict[str, Any]:
    """Evaluate model accuracy under repeated fault injection."""
    import itertools
    from src.training.trainer import evaluate as _eval

    trial_accs: List[float] = []

    for trial in range(n_trials):
        if seed is not None:
            torch.manual_seed(seed + trial)
            injector.seed = seed + trial

        eval_loader = (
            itertools.islice(loader, max_batches) if max_batches is not None else loader
        )

        injector.inject(model)
        try:
            metrics = _eval(model, eval_loader, device)
            trial_accs.append(metrics["acc"])
        finally:
            injector.restore(model)

    mean_acc = sum(trial_accs) / len(trial_accs)

    return {
        "mean_acc":    mean_acc,
        "std_acc":     _std(trial_accs),
        "min_acc":     min(trial_accs),
        "max_acc":     max(trial_accs),
        "degradation": (baseline_acc - mean_acc) if baseline_acc is not None else None,
        "trial_accs":  trial_accs,
    }


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
