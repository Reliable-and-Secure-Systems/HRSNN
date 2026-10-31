"""Dedicated comparison of all SNN learning algorithms on a fixed setup."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing  import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import run_experiment, resolve_config_path
from src.utils.logging   import get_logger

logger = get_logger("run_learning_rules")

ALL_RULES = [
    "surrogate_gradient",
    "bptt",
    "stdp",
    "eprop",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare all SNN learning rules on a fixed dataset + coding"
    )
    p.add_argument("--dataset",     default="config/datasets/mnist.yaml",
                   metavar="PATH")
    p.add_argument("--arch",        default="config/architectures/mlp_5layer.yaml",
                   metavar="PATH")
    p.add_argument("--coding",      default="config/coding_schemes/rate.yaml",
                   metavar="PATH")
    p.add_argument("--rules",       nargs="+", default=None,
                   metavar="RULE",
                   help=f"Learning rules to compare (default: all). "
                        f"Available: {ALL_RULES}")
    p.add_argument("--seeds",       nargs="+", type=int, default=[42],
                   metavar="SEED",
                   help="One or more seeds; results are averaged if multiple given")
    p.add_argument("--device",      default="auto")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--dry-run",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ds_path   = resolve_config_path(args.dataset, "dataset")
    arch_path = resolve_config_path(args.arch,    "arch")
    cod_path  = resolve_config_path(args.coding,  "coding")

    rules = args.rules if args.rules else ALL_RULES
    rules = [resolve_config_path(r, "learning") for r in rules]
    seeds = args.seeds

    ds_name   = Path(ds_path).stem
    cod_name  = Path(cod_path).stem
    total     = len(rules) * len(seeds)

    print(f"Learning-rule comparison")
    print(f"  dataset : {ds_name}")
    print(f"  coding  : {cod_name}")
    print(f"  rules   : {[Path(r).stem for r in rules]}")
    print(f"  seeds   : {seeds}")
    print(f"  total   : {total} run(s)")

    if args.dry_run:
        print("\n(dry-run — no experiments executed)")
        return

    per_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed:   list[str]                        = []
    idx       = 0

    for lr_path in rules:
        lr_name = Path(lr_path).stem
        for seed in seeds:
            idx += 1
            run_name = f"lr_cmp_{ds_name}_{cod_name}_{lr_name}_seed{seed}"
            print(f"\n[{idx}/{total}]  {lr_name}  seed={seed}")
            try:
                result = run_experiment(
                    ds_path     = ds_path,
                    arch_path   = arch_path,
                    coding_path = cod_path,
                    lr_path     = lr_path,
                    run_name    = run_name,
                    seed        = seed,
                    device_str  = args.device,
                    results_dir = args.results_dir,
                )
                per_rule[lr_name].append(result)
                print(f"  test_acc={result['test_acc']:.4f}  "
                      f"best_val={result['best_val_acc']:.4f}")
            except Exception as exc:
                logger.error("Run %s failed: %s", run_name, exc, exc_info=True)
                failed.append(run_name)

    _print_comparison(ds_name, cod_name, rules, per_rule, failed)


def _print_comparison(
    ds_name:  str,
    cod_name: str,
    rules:    list[str],
    per_rule: dict[str, list[dict[str, Any]]],
    failed:   list[str],
) -> None:
    import statistics

    print("\n" + "=" * 65)
    print("LEARNING RULE COMPARISON")
    print(f"Dataset: {ds_name}  |  Coding: {cod_name}")
    print("=" * 65)

    if not per_rule:
        print("No successful runs.")
    else:
        col_w   = max(len(r) for r in per_rule)
        multi   = any(len(v) > 1 for v in per_rule.values())

        if multi:
            header = f"  {'rule':<{col_w}}  {'test_acc mean':>13}  {'±std':>6}  {'best_val mean':>13}"
        else:
            header = f"  {'rule':<{col_w}}  {'test_acc':>8}  {'best_val_acc':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        def _mean(key: str, results: list) -> float:
            vals = [r.get(key, 0.0) for r in results]
            return statistics.mean(vals) if vals else 0.0

        sorted_rules = sorted(per_rule.items(),
                              key=lambda kv: _mean("test_acc", kv[1]),
                              reverse=True)

        for rule_name, results in sorted_rules:
            mean_acc = _mean("test_acc",     results)
            mean_val = _mean("best_val_acc", results)
            if multi:
                std_acc = statistics.stdev([r["test_acc"] for r in results]) \
                          if len(results) > 1 else 0.0
                print(f"  {rule_name:<{col_w}}  {mean_acc:>13.4f}  "
                      f"{std_acc:>6.4f}  {mean_val:>13.4f}")
            else:
                print(f"  {rule_name:<{col_w}}  {mean_acc:>8.4f}  {mean_val:>12.4f}")

        best_rule = sorted_rules[0][0]
        best_acc  = _mean("test_acc", sorted_rules[0][1])
        print(f"\nBest rule: {best_rule}  (mean test_acc={best_acc:.4f})")

    if failed:
        print(f"\nFailed runs: {failed}")
    print()


if __name__ == "__main__":
    main()
