"""Running statistics accumulator and per-phase epoch snapshot."""
from __future__ import annotations

from dataclasses import dataclass


class RunningStats:
    """Accumulates per-batch loss and accuracy during one training or eval pass."""

    def __init__(self):
        self._total_loss: float = 0.0
        self._correct:    int   = 0
        self._total:      int   = 0

    def update(self, loss: float, n_correct: int, n_total: int) -> None:
        self._total_loss += loss * n_total
        self._correct    += n_correct
        self._total      += n_total

    def compute(self) -> dict[str, float]:
        """Return {"loss": ..., "acc": ...} averaged over all seen samples."""
        n = max(self._total, 1)
        return {"loss": self._total_loss / n, "acc": self._correct / n}

    def reset(self) -> None:
        self._total_loss = 0.0
        self._correct    = 0
        self._total      = 0

    @property
    def n_samples(self) -> int:
        return self._total


@dataclass(frozen=True)
class EpochMetrics:
    """Immutable snapshot of metrics for one phase (train or val)."""

    loss: float
    acc:  float

    @classmethod
    def from_running(cls, stats: RunningStats) -> "EpochMetrics":
        d = stats.compute()
        return cls(loss=d["loss"], acc=d["acc"])

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Serialize to a flat dict."""
        return {f"{prefix}loss": self.loss, f"{prefix}acc": self.acc}

    @classmethod
    def from_dict(cls, d: dict, prefix: str = "") -> "EpochMetrics":
        return cls(
            loss=float(d[f"{prefix}loss"]),
            acc=float(d[f"{prefix}acc"]),
        )
