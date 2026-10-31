"""Single-axis ablation: vary one experiment axis, fix all others."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing  import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common    import run_experiment, resolve_config_path, all_configs
from src.tracking.multi_run import MultiRunTracker
from src.utils.logging      import get_logger

logger = get_logger("run_ablation")

AXES = ("dataset", "coding", "arch", "learning")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-axis SNN ablation study")
    p.add_argument("--axis",           required=True, choices=AXES,
                   help="Which axis to ablate (vary)")
    p.add_argument("--base-dataset",   default="config/datasets/mnist.yaml",
                   metavar="PATH")
    p.add_argument("--base-arch",      default="config/architectures/mlp_5layer.yaml",
                   metavar="PATH")
    p.add_argument("--base-coding",    default="config/coding_schemes/rate.yaml",
                   metavar="PATH")
    p.add_argument("--base-learning",  default="config/learning_rules/surrogate_gradient.yaml",
                   metavar="PATH")
    p.add_argument("--variants",       nargs="*", default=None,
                   metavar="PATH/NAME",
                   help="Explicit variants for the ablated axis; "
                        "omit to use all configs in that category")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--device",         default="auto")
    p.add_argument("--results-dir",    default="results/raw")
    p.add_argument("--dry-run",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    base = {
        "dataset":  resolve_config_path(args.base_dataset,  "dataset"),
        "arch":     resolve_config_path(args.base_arch,     "arch"),
        "coding":   resolve_config_path(args.base_coding,   "coding"),
        "learning": resolve_config_path(args.base_learning, "learning"),
    }

    axis = args.axis
    if args.variants:
        variants = [resolve_config_path(v, axis) for v in args.variants]
    else:
        variants = all_configs(axis)

    total = len(variants)
    print(f"Ablation axis: {axis}  |  {total} variant(s)  |  seed={args.seed}")
    for v in variants:
        print(f"  {Path(v).stem}")

    if args.dry_run:
        print("\n(dry-run — no experiments executed)")
        return

    results: list[dict[str, Any]] = []
    failed:  list[str]            = []

    for i, variant in enumerate(variants, 1):
        cfg = dict(base)
        cfg[axis] = variant

        variant_name = Path(variant).stem
        base_names   = {
            k: Path(v).stem for k, v in base.items() if k != axis
        }
        run_name = (
            f"ablation_{axis}_{variant_name}"
            f"_ds{Path(cfg['dataset']).stem}"
            f"_cod{Path(cfg['coding']).stem}"
            f"_lr{Path(cfg['learning']).stem}"
            f"_seed{args.seed}"
        )

        print(f"\n[{i}/{total}]  {axis}={variant_name}")
        try:
            result = run_experiment(
                ds_path     = cfg["dataset"],
                arch_path   = cfg["arch"],
                coding_path = cfg["coding"],
                lr_path     = cfg["learning"],
                run_name    = run_name,
                seed        = args.seed,
                device_str  = args.device,
                results_dir = args.results_dir,
            )
            results.append(result)
            print(f"  test_acc={result['test_acc']:.4f}  "
                  f"best_val={result['best_val_acc']:.4f}")
        except Exception as exc:
            logger.error("Variant %s failed: %s", variant_name, exc, exc_info=True)
            failed.append(variant_name)

    _print_summary(axis, base, results, failed)


def _print_summary(
    axis:    str,
    base:    dict[str, str],
    results: list[dict[str, Any]],
    failed:  list[str],
) -> None:
    print("\n" + "=" * 60)
    print(f"ABLATION RESULTS — axis: {axis}")
    base_desc = "  ".join(f"{k}={Path(v).stem}" for k, v in base.items() if k != axis)
    print(f"Fixed: {base_desc}")
    print("=" * 60)

    if not results:
        print("No successful runs.")
    else:
        ranked = sorted(results, key=lambda r: r.get("test_acc", 0), reverse=True)
        col_w  = max(len(r[axis]) for r in results)
        header = f"  {'variant':<{col_w}}  {'test_acc':>8}  {'val_acc':>8}"
        print(header)
        print("  " + "-" * (col_w + 20))
        for r in ranked:
            print(f"  {r[axis]:<{col_w}}  {r['test_acc']:>8.4f}  "
                  f"{r['best_val_acc']:>8.4f}")
        print(f"\nBest: {ranked[0][axis]}  test_acc={ranked[0]['test_acc']:.4f}")

    if failed:
        print(f"\nFailed: {failed}")
    print()


if __name__ == "__main__":
    main()
