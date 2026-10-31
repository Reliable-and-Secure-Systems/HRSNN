"""MultiRunTracker — aggregates completed runs and produces comparison tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class MultiRunTracker:
    """Scans a results directory and aggregates per-run summaries."""

    def __init__(self, results_dir: str | Path = "results/raw"):
        self._dir     = Path(results_dir)
        self._summary: List[Dict[str, Any]] = []


    def scan(self) -> "MultiRunTracker":
        """Re-scan results_dir and refresh the internal summary list."""
        self._summary = _scan_results(self._dir)
        return self

    @property
    def summary(self) -> List[Dict[str, Any]]:
        if not self._summary:
            self.scan()
        return self._summary


    def best_runs(
        self,
        metric:    str  = "test_acc",
        top_n:     int  = 5,
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return up to top_n runs sorted by metric (descending by default)."""
        valid = [r for r in self.summary if metric in r]
        valid.sort(key=lambda r: r[metric], reverse=not ascending)
        return valid[:top_n]

    def filter(self, **kwargs) -> List[Dict[str, Any]]:
        """Return runs whose top-level keys match all provided values."""
        result = []
        for r in self.summary:
            if all(r.get(k) == v for k, v in kwargs.items()):
                result.append(r)
        return result


    def comparison_table(
        self,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Return a plain-text ASCII table of all runs."""
        rows = self.summary
        if not rows:
            return "(no runs found in results directory)"

        if columns is None:
            columns = [
                "run_name", "dataset", "coding", "learning_rule",
                "best_val_acc", "test_acc", "n_epochs",
            ]

        display = []
        for r in rows:
            row = {}
            for col in columns:
                v = r.get(col)
                if v is None:
                    row[col] = "—"
                elif isinstance(v, float):
                    row[col] = f"{v:.4f}"
                else:
                    row[col] = str(v)
            display.append(row)

        widths = {
            col: max(len(col), max(len(d[col]) for d in display))
            for col in columns
        }

        sep    = "+-" + "-+-".join("-" * widths[c] for c in columns) + "-+"
        header = "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |"
        lines  = [sep, header, sep]
        for d in display:
            lines.append("| " + " | ".join(d[c].ljust(widths[c]) for c in columns) + " |")
        lines.append(sep)
        return "\n".join(lines)


    def to_csv(self, path: str | Path) -> None:
        """Write the summary to a CSV file (skips nested config dict)."""
        rows = self.summary
        if not rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        all_keys = sorted({k for r in rows for k in r if k != "config"})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def compare_runs(results_dir: str | Path = "results/raw") -> List[Dict]:
    """Scan results_dir and return a list of run summary dicts."""
    return _scan_results(Path(results_dir))


def _scan_results(results_dir: Path) -> List[Dict[str, Any]]:
    if not results_dir.exists():
        return []

    summary = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        entry: Dict[str, Any] = {"run_name": run_dir.name}

        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            entry["config"] = cfg
            for section, val in cfg.items():
                if isinstance(val, dict) and "name" in val:
                    entry[section] = val["name"]
            for section in ("learning", "learning_rule"):
                if section in cfg and isinstance(cfg[section], dict):
                    entry["learning_rule"] = cfg[section].get("name", "")
                    if "lr" in cfg[section]:
                        entry["lr"] = cfg[section]["lr"]

        test_path = run_dir / "test_metrics.json"
        if test_path.exists():
            with open(test_path) as f:
                entry.update(json.load(f))

        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists():
            with open(metrics_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                try:
                    entry["best_val_acc"] = max(float(r["val_acc"]) for r in rows)
                    entry["n_epochs"]     = len(rows)
                except (KeyError, ValueError):
                    pass

        summary.append(entry)

    return summary
