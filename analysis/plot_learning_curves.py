"""Plot validation accuracy (or any epoch metric) over training epochs."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib     import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    load_all_runs, load_epoch_history, group_runs,
    set_plot_style, savefig,
    SCHEME_COLORS, SCHEME_LABELS, SCHEME_ORDER,
    RULE_COLORS, RULE_LABELS, RULE_ORDER,
)


def plot_learning_curves(
    results_dir:    str        = "results/raw",
    group_by:       str        = "coding",
    metric:         str        = "val_acc",
    dataset_filter: str | None = None,
    smooth_window:  int        = 1,
    figsize:        tuple      = (9, 5),
    output:         str        = "results/figures/learning_curves.png",
) -> None:
    """Plot an epoch metric over training, one curve per group of runs."""
    import numpy as np
    import matplotlib.pyplot as plt

    set_plot_style()

    runs = load_all_runs(results_dir)
    if dataset_filter:
        runs = [r for r in runs if r.get("dataset") == dataset_filter]
    if not runs:
        print("No runs found.")
        return

    groups = group_runs(runs, by=group_by)

    if group_by == "coding":
        color_map = SCHEME_COLORS
        label_map = SCHEME_LABELS
        order     = [g for g in SCHEME_ORDER if g in groups] + \
                    [g for g in groups if g not in SCHEME_ORDER]
    elif group_by == "learning":
        color_map = RULE_COLORS
        label_map = RULE_LABELS
        order     = [g for g in RULE_ORDER if g in groups] + \
                    [g for g in groups if g not in RULE_ORDER]
    else:
        color_map = {}
        label_map = {}
        order     = sorted(groups.keys())

    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=figsize)

    for idx, gname in enumerate(order):
        group_runs_ = groups[gname]
        color       = color_map.get(gname, cmap(idx / max(len(order), 1)))
        label       = label_map.get(gname, gname)

        curves = []
        for run in group_runs_:
            hist = load_epoch_history(run["run_dir"])
            if not hist:
                continue
            vals = [row.get(metric, float("nan")) for row in hist]
            if smooth_window > 1:
                vals = _smooth(vals, smooth_window)
            curves.append(vals)

        if not curves:
            continue

        max_len = max(len(c) for c in curves)
        mat = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            mat[i, :len(c)] = c

        epochs = np.arange(1, max_len + 1)
        mean   = np.nanmean(mat, axis=0)
        std    = np.nanstd(mat,  axis=0)

        ax.plot(epochs, mean, color=color, label=label)
        if len(curves) > 1:
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=color)

    metric_label = metric.replace("_", " ").title()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    title_parts = [f"{metric_label} vs Epoch"]
    if dataset_filter:
        title_parts.append(f"({dataset_filter.upper()})")
    ax.set_title(" ".join(title_parts))
    ax.legend(loc="best")

    if "acc" in metric:
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}" if y <= 1.0 else f"{y:.1f}")
        )

    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)


def _smooth(values: list[float], window: int) -> list[float]:
    import numpy as np
    arr  = np.array(values, dtype=float)
    kern = np.ones(window) / window
    return list(np.convolve(arr, kern, mode="same"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot learning curves from run history")
    p.add_argument("--results-dir",    default="results/raw")
    p.add_argument("--group-by",       default="coding",
                   choices=["coding", "learning", "dataset", "arch"])
    p.add_argument("--metric",         default="val_acc",
                   help="Column from metrics.csv (val_acc, train_loss, …)")
    p.add_argument("--dataset",        default=None,
                   help="Filter to a specific dataset (mnist, cifar10)")
    p.add_argument("--smooth",         type=int, default=1,
                   help="Moving-average window size (1 = off)")
    p.add_argument("--output",         default="results/figures/learning_curves.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_learning_curves(
        results_dir    = args.results_dir,
        group_by       = args.group_by,
        metric         = args.metric,
        dataset_filter = args.dataset,
        smooth_window  = args.smooth,
        output         = args.output,
    )


if __name__ == "__main__":
    main()
