from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, List


def make_dirs(*paths: str | Path):
    """Create directories (and parents) for each path given."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str | Path, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_csv(rows: List[dict], path: str | Path):
    """Write a list of dicts to CSV. All dicts must share the same keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(row: dict, path: str | Path):
    """Append one row to a CSV. Creates the file with header if it does not exist."""
    path = Path(path)
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_csv(path: str | Path) -> List[dict]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))
