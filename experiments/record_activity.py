"""Backfill spike counts and SynOps for runs trained before they were recorded."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common         import load_all_runs, rebuild_pipeline
from src.tracking.activity    import measure_activity
from src.utils.io             import save_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill spike counts and SynOps per run")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--dataset",     default=None,
                   help="Filter by dataset name (mnist | cifar10)")
    p.add_argument("--device",      default="auto")
    p.add_argument("--overwrite",   action="store_true",
                   help="Re-measure runs that already have an activity block")
    p.add_argument("--dry-run",     action="store_true",
                   help="List planned work without running anything")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    runs = load_all_runs(args.results_dir)
    if args.dataset:
        runs = [r for r in runs if r.get("dataset") == args.dataset]
    runs = [
        r for r in runs
        if (Path(r["run_dir"]) / "best_model.pth").exists()
        and (Path(r["run_dir"]) / "test_metrics.json").exists()
        and not r.get("config", {}).get("synthetic", False)
    ]
    todo = [r for r in runs if args.overwrite or not r.get("activity")]

    print(f"Runs with checkpoint + test metrics: {len(runs)}")
    print(f"To measure                         : {len(todo)}\n")
    if args.dry_run:
        for r in runs:
            status = "pending" if r in todo else "HAS ACTIVITY"
            print(f"  {r['run_name']:<55}  [{status}]")
        return

    for i, run in enumerate(todo, 1):
        run_dir = Path(run["run_dir"])
        print(f"[{i}/{len(todo)}]  {run['run_name']}")
        model, loader, device, _ = rebuild_pipeline(run_dir, device_str=args.device)
        activity = {**measure_activity(model, loader, device), "source": "backfill"}

        path = run_dir / "test_metrics.json"
        with open(path) as f:
            metrics = json.load(f)
        metrics["activity"] = activity
        save_json(metrics, path)
        print(f"  syn_ops/sample={activity['syn_ops_per_sample']:.4g}"
              f"  spike_rate={activity['mean_spike_rate']:.4f}"
              f"  acc@T={activity['acc_by_timestep'][-1]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
