"""Checkpointer — save and reload best model weights during training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Checkpointer:
    """Saves model.state_dict() whenever a monitored metric improves."""

    FILENAME = "best_model.pth"

    def __init__(
        self,
        save_dir: Path | str,
        metric:   str = "val_acc",
        mode:     str = "max",
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self._dir    = Path(save_dir)
        self._metric = metric
        self._mode   = mode
        self._best   = float("-inf") if mode == "max" else float("inf")
        self._path   = self._dir / self.FILENAME


    def maybe_save(
        self,
        model:   nn.Module,
        epoch:   int,
        metrics: Dict[str, float],
    ) -> bool:
        """Conditionally save model weights."""
        value = float(metrics.get(self._metric, 0.0))
        if not self._is_better(value):
            return False

        self._best = value
        self._dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch":       epoch,
                self._metric:  value,
                "state_dict":  model.state_dict(),
            },
            self._path,
        )
        logger.info("Checkpoint  epoch=%d  %s=%.4f", epoch, self._metric, value)
        return True

    def load(self, model: nn.Module) -> Optional[Dict]:
        """Load the best checkpoint into model in-place."""
        if not self._path.exists():
            logger.warning("No checkpoint found at %s", self._path)
            return None
        ckpt = torch.load(self._path, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(
            "Loaded checkpoint  epoch=%d  %s=%.4f",
            ckpt.get("epoch", -1),
            self._metric,
            ckpt.get(self._metric, 0.0),
        )
        return ckpt


    @property
    def best(self) -> float:
        """Best metric value observed so far."""
        return self._best

    @property
    def path(self) -> Path:
        return self._path

    def _is_better(self, value: float) -> bool:
        return value > self._best if self._mode == "max" else value < self._best
