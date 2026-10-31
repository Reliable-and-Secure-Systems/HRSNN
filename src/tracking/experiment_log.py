"""EpochRecord dataclass and ExperimentLog — persistent ordered run history."""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterator, List


@dataclass
class EpochRecord:
    """Full record for one completed epoch (train + val + lr + timing)."""

    epoch:      int
    train_loss: float
    train_acc:  float
    val_loss:   float
    val_acc:    float
    lr:         float
    time_s:     float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpochRecord":
        field_names = {f.name for f in fields(cls)}
        coerce = {
            "epoch":      int,
            "train_loss": float, "train_acc": float,
            "val_loss":   float, "val_acc":   float,
            "lr":         float, "time_s":    float,
        }
        return cls(**{
            k: coerce.get(k, lambda x: x)(v)
            for k, v in d.items()
            if k in field_names
        })


class ExperimentLog:
    """Ordered list of EpochRecords with live CSV persistence."""

    _FIELDS: list[str] = [f.name for f in fields(EpochRecord)]

    def __init__(self, csv_path: Path | str):
        self._path    = Path(csv_path)
        self._records: List[EpochRecord] = []
        self._header_written: bool = (
            self._path.exists() and self._path.stat().st_size > 0
        )


    def append(self, record: EpochRecord) -> None:
        self._records.append(record)
        self._write_row(record)

    def best_val_acc(self) -> float:
        if not self._records:
            return 0.0
        return max(r.val_acc for r in self._records)

    def best_epoch(self) -> int:
        """Return the epoch number with the highest val_acc."""
        if not self._records:
            return 0
        return max(self._records, key=lambda r: r.val_acc).epoch

    @property
    def records(self) -> List[EpochRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[EpochRecord]:
        return iter(self._records)


    def _write_row(self, record: EpochRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDS)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(record.to_dict())

    @classmethod
    def load(cls, csv_path: Path | str) -> "ExperimentLog":
        """Reconstruct an ExperimentLog from an existing metrics CSV."""
        log = cls(csv_path)
        if not log._path.exists():
            return log
        with open(log._path, newline="") as f:
            for row in csv.DictReader(f):
                log._records.append(EpochRecord.from_dict(row))
        log._header_written = True
        return log
