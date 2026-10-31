"""ExperimentTracker — coordinates logging, checkpointing, and progress display."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn

from src.tracking.experiment_log import EpochRecord, ExperimentLog
from src.tracking.checkpointer   import Checkpointer
from src.utils.io                import save_json
from src.utils.logging           import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Single-run coordinator: wraps ExperimentLog + Checkpointer and prints formatted progress after each epoch."""

    def __init__(
        self,
        run_name:    str,
        config:      Dict[str, Any],
        results_dir: str | Path = "results/raw",
    ):
        self.run_name = run_name
        self.run_dir  = Path(results_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        for stale in ("metrics.csv", "test_metrics.json", "fault_results.json"):
            path = self.run_dir / stale
            if path.exists():
                path.unlink()
        metrics_csv = self.run_dir / "metrics.csv"
        self._log  = ExperimentLog(metrics_csv)
        self._ckpt = Checkpointer(self.run_dir)

        save_json(config, self.run_dir / "config.json")
        logger.info("Run '%s'  ->  %s", run_name, self.run_dir)


    def on_epoch_end(self, row: Dict[str, float], model: nn.Module) -> None:
        """Log, checkpoint, and print progress for one completed epoch."""
        record = EpochRecord(
            epoch      = int(row["epoch"]),
            train_loss = float(row["train_loss"]),
            train_acc  = float(row["train_acc"]),
            val_loss   = float(row["val_loss"]),
            val_acc    = float(row["val_acc"]),
            lr         = float(row.get("lr",     0.0)),
            time_s     = float(row.get("time_s", 0.0)),
        )
        self._log.append(record)
        self._ckpt.maybe_save(model, record.epoch, row)

    def log_final(self, test_metrics: Dict[str, Any]) -> None:
        """Persist test-set metrics to test_metrics.json (nested blocks allowed)."""
        save_json(test_metrics, self.run_dir / "test_metrics.json")
        logger.info(
            "Test  %s",
            "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()
                      if isinstance(v, (int, float))),
        )


    def best_val_acc(self) -> float:
        return self._log.best_val_acc()

    def best_epoch(self) -> int:
        return self._log.best_epoch()

    def load_best_checkpoint(self, model: nn.Module) -> Optional[Dict]:
        """Load best weights into model in-place; returns checkpoint dict."""
        return self._ckpt.load(model)

    @property
    def history(self):
        """List of EpochRecord objects for all logged epochs."""
        return self._log.records


    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        record = EpochRecord(
            epoch      = epoch,
            train_loss = float(metrics.get("train_loss", 0.0)),
            train_acc  = float(metrics.get("train_acc",  0.0)),
            val_loss   = float(metrics.get("val_loss",   0.0)),
            val_acc    = float(metrics.get("val_acc",    0.0)),
            lr         = float(metrics.get("lr",         0.0)),
            time_s     = float(metrics.get("time_s",     0.0)),
        )
        self._log.append(record)

    def save_checkpoint(
        self, model: nn.Module, epoch: int, val_acc: float
    ) -> bool:
        return self._ckpt.maybe_save(model, epoch, {"val_acc": val_acc})
