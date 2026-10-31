"""Non-parametric statistical tests for comparing SNN experiment groups."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib   import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    load_all_runs, group_runs,
    SCHEME_LABELS, RULE_LABELS,
)


def run_statistical_tests(
    results_dir:    str        = "results/raw",
    group_by:       str        = "coding",
    metric:         str        = "test_acc",
    dataset_filter: str | None = None,
    alpha:          float      = 0.05,
    output:         str        = "results/aggregated/statistical_tests.json",
) -> dict:
    """Run Friedman + pairwise Wilcoxon tests on collected experiment results."""
    from scipy import stats

    runs = load_all_runs(results_dir)
    if dataset_filter:
        runs = [r for r in runs if r.get("dataset") == dataset_filter]

    groups = group_runs(runs, by=group_by)

    label_map = {**SCHEME_LABELS, **RULE_LABELS}

    factors = [f for f in BLOCK_FACTORS if f != group_by]
    group_blocks: dict[str, dict[tuple, float]] = {}
    for gname, group_runs_ in groups.items():
        block_map: dict[tuple, float] = {}
        for r in group_runs_:
            val = r.get(metric)
            if val is None:
                continue
            block = tuple(r.get(f) for f in factors)
            if block in block_map:
                raise ValueError(f"Two runs of group '{gname}' share block "
                                 f"{dict(zip(factors, block))}; cannot match them.")
            block_map[block] = float(val)
        if block_map:
            group_blocks[gname] = block_map

    common_blocks = sorted(
        set.intersection(*[set(b.keys()) for b in group_blocks.values()])
        if group_blocks else set(), key=str,
    )
    if not common_blocks:
        print("No block is present in every group — cannot run matched tests.")
        return {}
    print(f"Matched blocks: {len(common_blocks)} (factors held fixed: {', '.join(factors)})")

    group_values: dict[str, list[float]] = {
        g: [block_map[b] for b in common_blocks]
        for g, block_map in group_blocks.items()
    }

    if len(group_values) < 2:
        print("Need at least 2 groups with data.")
        return {}

    results: dict = {
        "metric":         metric,
        "group_by":       group_by,
        "dataset_filter": dataset_filter,
        "alpha":          alpha,
        "block_factors":  factors,
        "n_blocks":       len(common_blocks),
        "groups":         {
            g: {"n": len(v), "mean": _mean(v), "std": _std(v), "values": v}
            for g, v in group_values.items()
        },
    }

    print(f"\n{'='*60}")
    print(f"Statistical Tests  |  metric={metric}  |  group_by={group_by}"
          + (f"  |  dataset={dataset_filter}" if dataset_filter else ""))
    print(f"{'='*60}")
    print(f"\n{'Group':<28}  {'n':>4}  {'mean':>8}  {'std':>7}")
    print("-" * 52)
    for g, v in sorted(group_values.items(), key=lambda kv: -_mean(kv[1])):
        print(f"{label_map.get(g, g):<28}  {len(v):>4}  {_mean(v):>8.4f}  {_std(v):>7.4f}")

    sizes = [len(v) for v in group_values.values()]
    min_size = min(sizes)
    friedman_result = None

    if len(group_values) >= 3 and min_size >= 3:
        truncated = [v[:min_size] for v in group_values.values()]
        try:
            stat, p = stats.friedmanchisquare(*truncated)
            friedman_result = {"statistic": stat, "p_value": p, "n_per_group": min_size}
            sig = "✓ significant" if p < alpha else "✗ not significant"
            print(f"\nFriedman test  χ²={stat:.3f}  p={p:.4f}  [{sig} at α={alpha}]")
        except Exception as e:
            print(f"\nFriedman test failed: {e}")
    else:
        print(f"\nFriedman test skipped (need ≥3 groups with ≥3 samples each; "
              f"got {len(group_values)} groups, min n={min_size})")

    results["friedman"] = friedman_result

    print(f"\nPairwise Wilcoxon signed-rank  (α={alpha})")
    pairs = list(combinations(sorted(group_values.keys()), 2))
    wilcoxon_results = {}

    for g1, g2 in pairs:
        v1, v2 = group_values[g1], group_values[g2]
        n       = min(len(v1), len(v2))

        if n < 5:
            print(f"  {label_map.get(g1,g1)!s:<22} vs {label_map.get(g2,g2)!s:<22}"
                  f"  — skipped (n={n} < 5)")
            continue

        v1t, v2t = v1[:n], v2[:n]
        try:
            stat, p = stats.wilcoxon(v1t, v2t, alternative="two-sided")
            r = _signed_rank_biserial(v1t, v2t)
            effect = _effect_label(abs(r))
            wilcoxon_results[f"{g1}__vs__{g2}"] = {
                "statistic": stat, "p_value": p,
                "effect_r": r, "effect_label": effect, "n": n,
            }
        except Exception as e:
            print(f"  {g1} vs {g2}: {e}")

    for key, p_holm in _holm(
            {k: v["p_value"] for k, v in wilcoxon_results.items()}).items():
        wilcoxon_results[key]["p_holm"] = p_holm
    for key, res in wilcoxon_results.items():
        g1, g2 = key.split("__vs__")
        sig = "*" if res["p_holm"] < alpha else " "
        print(f"  {label_map.get(g1,g1)!s:<22} vs {label_map.get(g2,g2)!s:<22}"
              f"  p={res['p_value']:.4f}  p_holm={res['p_holm']:.4f}{sig}"
              f"  r={res['effect_r']:+.3f} ({res['effect_label']})")

    results["wilcoxon_pairwise"] = wilcoxon_results
    results["multiple_comparisons"] = "holm"

    try:
        import scikit_posthocs as sp
        import pandas as pd

        if friedman_result and friedman_result["p_value"] < alpha:
            df = pd.DataFrame({g: v[:min_size] for g, v in group_values.items()})
            nemenyi = sp.posthoc_nemenyi_friedman(df)
            print(f"\nNemenyi post-hoc p-values:")
            print(nemenyi.round(4).to_string())
            results["nemenyi"] = nemenyi.to_dict()
    except ImportError:
        pass

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x),
        encoding="utf-8",
    )
    print(f"\nResults saved: {out}")

    _print_pvalue_matrix(wilcoxon_results, group_values.keys(), label_map, alpha)

    return results


BLOCK_FACTORS = ("dataset", "coding", "learning", "arch", "seed")


def _signed_rank_biserial(v1: list[float], v2: list[float]) -> float:
    """(W+ - W-) / (W+ + W-) over the nonzero paired differences v1 - v2."""
    from scipy.stats import rankdata
    d = [a - b for a, b in zip(v1, v2) if a != b]
    if not d:
        return 0.0
    ranks = rankdata([abs(x) for x in d])
    w_pos = sum(r for r, x in zip(ranks, d) if x > 0)
    w_neg = sum(r for r, x in zip(ranks, d) if x < 0)
    return (w_pos - w_neg) / (w_pos + w_neg)


def _holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for one family of tests."""
    order = sorted(p_values, key=p_values.get)
    m = len(order)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0

def _std(v: list[float]) -> float:
    import statistics
    return statistics.stdev(v) if len(v) > 1 else 0.0

def _effect_label(r: float) -> str:
    if r < 0.1:  return "negligible"
    if r < 0.3:  return "small"
    if r < 0.5:  return "medium"
    return "large"

def _print_pvalue_matrix(
    wilcoxon_results: dict,
    group_names,
    label_map: dict,
    alpha: float,
) -> None:
    names = sorted(group_names)
    if not names or len(names) < 2:
        return

    labels = [label_map.get(n, n)[:14] for n in names]
    w      = max(len(l) for l in labels)

    print(f"\nHolm-adjusted p-value matrix  (* = p_holm < {alpha})")
    header = " " * (w + 2) + "".join(f"  {l:>{w}}" for l in labels)
    print(header)

    for i, g1 in enumerate(names):
        row = f"{labels[i]:>{w}}  "
        for j, g2 in enumerate(names):
            if i == j:
                row += f"  {'—':>{w}}"
            else:
                key1 = f"{g1}__vs__{g2}"
                key2 = f"{g2}__vs__{g1}"
                entry = wilcoxon_results.get(key1) or wilcoxon_results.get(key2)
                if entry:
                    p   = entry["p_holm"]
                    sig = "*" if p < alpha else " "
                    row += f"  {p:.3f}{sig}".rjust(w + 2)
                else:
                    row += f"  {'n/a':>{w}}"
        print(row)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Friedman + pairwise Wilcoxon tests on experiment results"
    )
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--group-by",    default="coding",
                   choices=["coding", "learning", "dataset"])
    p.add_argument("--metric",      default="test_acc",
                   help="Metric to test (test_acc, test_loss, best_val_acc)")
    p.add_argument("--dataset",     default=None,
                   help="Restrict to one dataset")
    p.add_argument("--alpha",       type=float, default=0.05,
                   help="Significance level (default 0.05)")
    p.add_argument("--output",
                   default="results/aggregated/statistical_tests.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_statistical_tests(
        results_dir    = args.results_dir,
        group_by       = args.group_by,
        metric         = args.metric,
        dataset_filter = args.dataset,
        alpha          = args.alpha,
        output         = args.output,
    )


if __name__ == "__main__":
    main()
