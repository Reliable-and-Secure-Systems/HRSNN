"""Horizontal bar chart ranking groups (coding schemes, learning rules) by a metric."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    load_all_runs, group_runs,
    set_plot_style, savefig,
    SCHEME_COLORS, SCHEME_LABELS, SCHEME_ORDER,
    RULE_COLORS, RULE_LABELS, RULE_ORDER,
)


def plot_comparison_bars(
    results_dir:    str        = "results/raw",
    metric:         str        = "test_acc",
    group_by:       str        = "coding",
    dataset_filter: str | None = None,
    facet_by:       str | None = None,
    figsize:        tuple      = (8, 5),
    output:         str        = "results/figures/comparison_bars.png",
) -> None:
    """Plot a horizontal bar chart ranking groups of runs by a metric."""
    import numpy as np
    import matplotlib.pyplot as plt

    set_plot_style()

    runs = load_all_runs(results_dir)
    if dataset_filter:
        runs = [r for r in runs if r.get("dataset") == dataset_filter]
    if not runs:
        print("No runs found.")
        return

    if group_by == "coding":
        color_map = SCHEME_COLORS
        label_map = SCHEME_LABELS
        order     = SCHEME_ORDER
    elif group_by == "learning":
        color_map = RULE_COLORS
        label_map = RULE_LABELS
        order     = RULE_ORDER
    else:
        color_map, label_map, order = {}, {}, []

    if facet_by:
        facets = sorted({r.get(facet_by, "?") for r in runs})
        fig, axes = plt.subplots(1, len(facets),
                                 figsize=(figsize[0] * len(facets), figsize[1]),
                                 sharey=True)
        if len(facets) == 1:
            axes = [axes]
        for ax, fval in zip(axes, facets):
            subset = [r for r in runs if r.get(facet_by) == fval]
            _draw_bars(ax, subset, metric, group_by, color_map, label_map, order)
            ax.set_title(str(fval).upper())
        fig.suptitle(f"{metric.replace('_', ' ').title()} by {group_by}", y=1.02)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        _draw_bars(ax, runs, metric, group_by, color_map, label_map, order)
        ax.set_title(f"{metric.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}")

    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)


def _draw_bars(ax, runs, metric, group_by, color_map, label_map, order):
    import numpy as np
    import matplotlib.pyplot as plt

    groups  = group_runs(runs, by=group_by)
    present = [g for g in order if g in groups] + \
              [g for g in sorted(groups) if g not in order]

    means, stds, colors, labels = [], [], [], []
    for gname in present:
        vals = [r.get(metric) for r in groups[gname] if r.get(metric) is not None]
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        colors.append(color_map.get(gname, "#888888"))
        labels.append(label_map.get(gname, gname))

    if not means:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    order_idx = sorted(range(len(means)), key=lambda i: means[i])
    means  = [means[i]  for i in order_idx]
    stds   = [stds[i]   for i in order_idx]
    colors = [colors[i] for i in order_idx]
    labels = [labels[i] for i in order_idx]

    y_pos  = range(len(means))
    bars   = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85,
                     error_kw={"capsize": 4, "elinewidth": 1.5}, height=0.55)

    for bar, m in zip(bars, means):
        ax.text(m + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric.replace("_", " ").title())
    if "acc" in metric:
        ax.set_xlim(0, min(1.05, max(means) * 1.15))
    ax.axvline(x=max(means), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Horizontal bar comparison chart")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--metric",      default="test_acc",
                   help="Metric to rank (test_acc, test_loss, best_val_acc)")
    p.add_argument("--group-by",    default="coding",
                   choices=["coding", "learning", "dataset", "arch"])
    p.add_argument("--dataset",     default=None)
    p.add_argument("--facet-by",    default=None,
                   help="Facet subplots by this key (e.g. 'dataset')")
    p.add_argument("--output",      default="results/figures/comparison_bars.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_comparison_bars(
        results_dir    = args.results_dir,
        metric         = args.metric,
        group_by       = args.group_by,
        dataset_filter = args.dataset,
        facet_by       = args.facet_by,
        output         = args.output,
    )


if __name__ == "__main__":
    main()
