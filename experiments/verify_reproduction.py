"""Check re-trained runs against an archived copy of an earlier grid."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

EPOCH_FIELDS = ("train_loss", "train_acc", "val_loss", "val_acc")
CONFIG_SECTIONS = ("dataset", "coding", "arch", "learning")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify re-trained runs against an archive")
    p.add_argument("--archive", required=True,
                   help="Directory holding the archived run folders")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--quiet", action="store_true", help="Print only the summary")
    return p.parse_args()


def _epochs(run_dir: Path) -> list[tuple]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [tuple(row[k] for k in EPOCH_FIELDS) for row in csv.DictReader(f)]


def _test_metrics(run_dir: Path) -> tuple | None:
    path = run_dir / "test_metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return (data.get("test_acc"), data.get("test_loss"))


def compare(new_dir: Path, old_dir: Path) -> list[str]:
    """Differences between a re-trained run and its archived copy ([] = identical)."""
    diffs: list[str] = []

    new_cfg = json.loads((new_dir / "config.json").read_text())
    old_cfg = json.loads((old_dir / "config.json").read_text())
    for section in CONFIG_SECTIONS:
        if new_cfg.get(section) != old_cfg.get(section):
            diffs.append(f"config.{section}")
    if new_cfg.get("seed") != old_cfg.get("seed"):
        diffs.append("seed")

    new_ep, old_ep = _epochs(new_dir), _epochs(old_dir)
    if len(new_ep) != len(old_ep):
        diffs.append(f"epoch count {len(new_ep)} vs {len(old_ep)}")
    elif new_ep != old_ep:
        first = next(i + 1 for i, (a, b) in enumerate(zip(new_ep, old_ep)) if a != b)
        diffs.append(f"epochs differ from epoch {first}")

    if _test_metrics(new_dir) != _test_metrics(old_dir):
        diffs.append("test metrics")
    return diffs


def main() -> None:
    args = _parse_args()
    archive = Path(args.archive)
    results = Path(args.results_dir)
    if not archive.is_dir():
        sys.exit(f"Archive not found: {archive}")

    identical, differing, retrained, reused, only_new = [], [], [], [], []
    in_progress: list[str] = []

    for run_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        if not (run_dir / "config.json").exists():
            continue
        if not (run_dir / "test_metrics.json").exists():
            in_progress.append(run_dir.name)
            continue
        has_env = "environment" in json.loads((run_dir / "config.json").read_text())
        (retrained if has_env else reused).append(run_dir.name)

        old_dir = archive / run_dir.name
        if not (old_dir / "config.json").exists():
            only_new.append(run_dir.name)
            continue
        if not has_env:
            continue
        diffs = compare(run_dir, old_dir)
        (identical if not diffs else differing).append((run_dir.name, diffs))

    if not args.quiet:
        for name, _ in identical:
            print(f"  IDENTICAL  {name}")
        for name, diffs in differing:
            print(f"  DIFFERS    {name}: {', '.join(diffs)}")
        for name in only_new:
            print(f"  NEW        {name} (not in the archive)")
        for name in in_progress:
            print(f"  RUNNING    {name} (no test metrics yet, skipped)")

    print(f"\nre-trained runs         : {len(retrained)}")
    print(f"  reproduce the archive : {len(identical)}")
    print(f"  differ                : {len(differing)}")
    print(f"  not in the archive    : {len(only_new)}")
    print(f"archived runs reused    : {len(reused)}")
    if in_progress:
        print(f"still training, skipped : {len(in_progress)}")

    if differing:
        print("\nA differing run means today's code no longer reproduces that archived "
              "run, so the reused runs cannot be trusted either — re-train them.")
        sys.exit(1)
    print("\nEvery re-trained run reproduces its archived copy exactly.")


if __name__ == "__main__":
    main()
