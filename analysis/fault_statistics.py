"""Statistics on fault resilience: which coding scheme / learning rule loses the least accuracy under each fault model?"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import SCHEME_LABELS, RULE_LABELS, savefig, set_plot_style
from analysis.statistical_tests import (
    BLOCK_FACTORS, _effect_label, _holm, _mean, _signed_rank_biserial, _std,
)
from experiments.run_fault_evaluation import FAULT_MODEL_VERSION

ALPHA = 0.05
SCOPES = ("pooled", "mnist", "cifar10")
GROUPS = ("coding", "learning")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistics on fault resilience")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--summary", choices=("mean", "top"), default="mean")
    return p.parse_args()


def load_losses(results_dir: str) -> tuple[list[dict], dict]:
    """One record per (run, sweep): run factors + mean/top normalised loss."""
    records, sweep_labels = [], {}
    for path in sorted(Path(results_dir).glob("*/fault_results.json")):
        d = json.loads(path.read_text())
        cfg = json.loads((path.parent / "config.json").read_text())
        chance = 1.0 / cfg.get("n_classes", 10)
        base = {"dataset": d["dataset"], "coding": d["coding"], "learning": d["learning"],
                "arch": d.get("arch", "mlp"), "seed": d["seed"], "run": d["run_name"]}
        for name, sw in d["fault_sweeps"].items():
            if sw.get("fault_model_version") != FAULT_MODEL_VERSION:
                raise ValueError(f"{path}: sweep {name} has fault-model version "
                                 f"{sw.get('fault_model_version')}, expected {FAULT_MODEL_VERSION}")
            room = sw["substrate_acc"] - chance
            losses = [(sw["substrate_acc"] - r["mean_acc"]) / room for r in sw["results"]]
            sweep_labels[name] = sw.get("label", name)
            records.append({**base, "sweep": name,
                            "mean": sum(losses) / len(losses), "top": losses[-1]})
    return records, sweep_labels


def test_group(records: list[dict], group_by: str, summary: str) -> dict | None:
    """Friedman + Holm-corrected pairwise Wilcoxon on matched blocks."""
    from scipy import stats

    factors = [f for f in BLOCK_FACTORS if f != group_by]
    blocks: dict[str, dict[tuple, float]] = {}
    for r in records:
        key = tuple(r[f] for f in factors)
        gb = blocks.setdefault(r[group_by], {})
        if key in gb:
            raise ValueError(f"duplicate block {key} in group {r[group_by]}")
        gb[key] = r[summary]
    common = sorted(set.intersection(*[set(b) for b in blocks.values()]), key=str)
    if len(blocks) < 3 or len(common) < 5:
        return None
    vals = {g: [b[k] for k in common] for g, b in blocks.items()}
    fr_stat, fr_p = stats.friedmanchisquare(*vals.values())
    pairs = {}
    for g1, g2 in combinations(sorted(vals), 2):
        _, p = stats.wilcoxon(vals[g1], vals[g2], alternative="two-sided")
        r = _signed_rank_biserial(vals[g1], vals[g2])
        pairs[f"{g1}__vs__{g2}"] = {"p_value": p, "effect_r": r,
                                    "effect_label": _effect_label(abs(r))}
    for k, ph in _holm({k: v["p_value"] for k, v in pairs.items()}).items():
        pairs[k]["p_holm"] = ph
    return {"n_blocks": len(common), "block_factors": factors,
            "groups": {g: {"mean": _mean(v), "std": _std(v)} for g, v in vals.items()},
            "friedman": {"statistic": fr_stat, "p_value": fr_p},
            "wilcoxon_pairwise": pairs}


def significance_marks(res: dict) -> dict[str, str]:
    """'·' if a group is significantly less damaged than some group, '×' if more."""
    marks = {g: "" for g in res["groups"]}
    if res["friedman"]["p_value"] >= ALPHA:
        return marks
    for key, v in res["wilcoxon_pairwise"].items():
        if v["p_holm"] >= ALPHA:
            continue
        g1, g2 = key.split("__vs__")
        better, worse = (g2, g1) if v["effect_r"] > 0 else (g1, g2)
        marks[better] += "" if "·" in marks[better] else "·"
        marks[worse] += "" if "×" in marks[worse] else "×"
    return marks


def plot(results: dict, sweep_labels: dict, group_by: str, scope: str, summary: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    set_plot_style()
    sweeps = [s for s in sweep_labels if results.get(s, {}).get(group_by, {}).get(scope)]
    if not sweeps:
        return
    groups = sorted(results[sweeps[0]][group_by][scope]["groups"])
    labels = {**SCHEME_LABELS, **RULE_LABELS}
    m = np.array([[results[s][group_by][scope]["groups"][g]["mean"] for s in sweeps] for g in groups])
    fig, ax = plt.subplots(figsize=(1.15 * len(sweeps) + 2.5, 0.55 * len(groups) + 2.2))
    im = ax.imshow(m, cmap="Reds", vmin=0, vmax=max(0.5, float(m.max())), aspect="auto")
    for j, s in enumerate(sweeps):
        marks = significance_marks(results[s][group_by][scope])
        for i, g in enumerate(groups):
            ax.text(j, i, f"{m[i, j]:.2f}{marks[g]}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.6 * max(0.5, float(m.max())) else "black")
    ax.set_xticks(range(len(sweeps)))
    ax.set_xticklabels([sweep_labels[s] for s in sweeps], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([labels.get(g, g) for g in groups])
    what = "averaged over the swept fault rates" if summary == "mean" else "at the highest fault rate"
    ax.set_title(f"Fraction of above-chance accuracy lost, {what}\n"
                 f"[{scope}] · = significantly less damaged, × = more (Holm, α={ALPHA})",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    suffix = "" if summary == "mean" else "_top"
    savefig(fig, f"results/figures/fault_robustness_{group_by}_{scope}{suffix}.png")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    records, sweep_labels = load_losses(args.results_dir)
    results: dict = {}
    rows = []
    for sweep in sweep_labels:
        for group_by in GROUPS:
            for scope in SCOPES:
                recs = [r for r in records if r["sweep"] == sweep
                        and (scope == "pooled" or r["dataset"] == scope)]
                res = test_group(recs, group_by, args.summary)
                if res is None:
                    continue
                results.setdefault(sweep, {}).setdefault(group_by, {})[scope] = res
                order = sorted(res["groups"], key=lambda g: res["groups"][g]["mean"])
                sig = [k for k, v in res["wilcoxon_pairwise"].items() if v["p_holm"] < ALPHA]
                rows.append({
                    "sweep": sweep, "group_by": group_by, "scope": scope,
                    "n_blocks": res["n_blocks"],
                    "friedman_p": f"{res['friedman']['p_value']:.2e}",
                    "least_to_most_damaged": " < ".join(
                        f"{g} {res['groups'][g]['mean']:.3f}" for g in order),
                    "significant_pairs_holm": "; ".join(
                        f"{k} (r={res['wilcoxon_pairwise'][k]['effect_r']:+.2f})" for k in sig),
                })
    suffix = "" if args.summary == "mean" else "_top"
    out = Path(f"results/aggregated/fault_statistics{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"measure": "(substrate_acc - acc) / (substrate_acc - chance)",
                               "summary": args.summary, "alpha": ALPHA,
                               "fault_model_version": FAULT_MODEL_VERSION,
                               "sweeps": results},
                              indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x))
    with open(f"results/aggregated/fault_statistics_summary{suffix}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for group_by in GROUPS:
        for scope in SCOPES:
            plot(results, sweep_labels, group_by, scope, args.summary)
    print(f"{len(rows)} test families -> {out}")


if __name__ == "__main__":
    main()
