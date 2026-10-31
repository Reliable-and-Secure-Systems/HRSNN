"""Full grid sweep over any combination of datasets × coding schemes × learning rules."""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib  import Path
from typing   import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common  import run_experiment, resolve_config_path, all_configs
from src.tracking.multi_run import MultiRunTracker
from src.utils.logging      import get_logger

logger = get_logger("run_sweep")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep over SNN experiment axes")

    p.add_argument("--datasets",  nargs="+", default=["mnist"],
                   metavar="DATASET",
                   help="Dataset configs: paths, short names, or 'all'")
    p.add_argument("--archs",     nargs="+", default=["mlp_5layer"],
                   metavar="ARCH",
                   help="Architecture configs: paths, short names, or 'all'")
    p.add_argument("--codings",   nargs="+", default=["rate"],
                   metavar="CODING",
                   help="Coding-scheme configs: paths, short names, or 'all'")
    p.add_argument("--learners",  nargs="+", default=["surrogate_gradient"],
                   metavar="LEARNER",
                   help="Learning-rule configs: paths, short names, or 'all'")
    p.add_argument("--seeds",     nargs="+", type=int, default=[42],
                   metavar="SEED")
    p.add_argument("--device",    default="auto")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs that already wrote test_metrics.json (resume)")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print all combinations without running any experiment")
    return p.parse_args()


def _is_cuda_context_loss(exc: BaseException) -> bool:
    """Whether an exception means this process's CUDA context is unusable."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda error" in text or "cuda_error" in text or "acceleratorerror" in text


def _run_name(ds: str, arch: str, cod: str, lr: str, seed: int) -> str:
    """Run folder name; the architecture is not part of it (one arch in the grid)."""
    return f"{Path(ds).stem}_{Path(cod).stem}_{Path(lr).stem}_seed{seed}"


def _resolve_list(names: list[str], category: str) -> list[str]:
    """Expand 'all' to every YAML in the category dir; resolve each name."""
    if names == ["all"]:
        return all_configs(category)
    return [resolve_config_path(n, category) for n in names]


def main() -> None:
    args = _parse_args()

    datasets  = _resolve_list(args.datasets,  "dataset")
    archs     = _resolve_list(args.archs,     "arch")
    codings   = _resolve_list(args.codings,   "coding")
    learners  = _resolve_list(args.learners,  "learning")
    seeds     = args.seeds

    combos = list(product(datasets, archs, codings, learners, seeds))
    total  = len(combos)

    done = {
        combo for combo in combos
        if args.skip_existing
        and (Path(args.results_dir) / _run_name(*combo) / "test_metrics.json").exists()
    }

    print(f"Sweep: {total} combinations"
          + (f"  ({len(done)} already finished, skipped)" if done else ""))
    for i, combo in enumerate(combos, 1):
        ds, arch, cod, lr, seed = combo
        label = (f"  [{i:>{len(str(total))}}/{total}]"
                 f"  ds={Path(ds).stem}"
                 f"  coding={Path(cod).stem}"
                 f"  lr={Path(lr).stem}"
                 f"  seed={seed}"
                 + ("  [done]" if combo in done else ""))
        print(label)

    if args.dry_run:
        print("\n(dry-run — no experiments executed)")
        return

    results: list[dict[str, Any]] = []
    failed:  list[str]            = []
    aborted: str | None           = None

    for i, combo in enumerate(combos, 1):
        if combo in done:
            continue
        ds, arch, cod, lr, seed = combo
        run_name = _run_name(*combo)
        print(f"\n[{i}/{total}]  {run_name}")
        try:
            result = run_experiment(
                ds_path     = ds,
                arch_path   = arch,
                coding_path = cod,
                lr_path     = lr,
                run_name    = run_name,
                seed        = seed,
                device_str  = args.device,
                results_dir = args.results_dir,
            )
            results.append(result)
            print(f"  test_acc={result['test_acc']:.4f}  "
                  f"best_val={result['best_val_acc']:.4f}")
        except Exception as exc:
            logger.error("Run %s failed: %s", run_name, exc, exc_info=True)
            failed.append(run_name)
            if _is_cuda_context_loss(exc):
                aborted = run_name
                break

    _print_summary(results, failed, args.results_dir)

    if aborted:
        remaining = sum(1 for c in combos if c not in done) - len(results) - len(failed)
        print(f"STOPPED: the CUDA context was lost during {aborted} "
              f"(a driver update mid-run will do this).\n"
              f"{remaining} run(s) were not attempted. Re-run the same command "
              f"with --skip-existing to resume in a fresh process.\n")
    if failed:
        sys.exit(1)


def _print_summary(
    results:     list[dict[str, Any]],
    failed:      list[str],
    results_dir: str,
) -> None:
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)

    if not results:
        print("No successful runs.")
    else:
        mt = MultiRunTracker(results_dir)
        mt.scan()
        print(mt.comparison_table(
            columns=["run_name", "dataset", "coding", "learning",
                     "best_val_acc", "test_acc"]
        ))
        ranked = sorted(results, key=lambda r: r.get("test_acc", 0), reverse=True)
        print(f"\nBest run: {ranked[0]['run_name']}"
              f"  test_acc={ranked[0]['test_acc']:.4f}")

    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for name in failed:
            print(f"  {name}")

    print()


if __name__ == "__main__":
    main()
