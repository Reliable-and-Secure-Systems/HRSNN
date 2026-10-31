"""Fault retention curves — stacked (2x2) layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    set_plot_style, savefig,
    SCHEME_COLORS, SCHEME_LABELS, SCHEME_ORDER,
)
from analysis.plot_fault_reliability import (
    FAULT_ORDER, FAULT_LABELS, FAULT_SWEEPS, CATEGORIES,
    load_fault_results, select_category, aggregate_retention, _param_label,
)


def plot_retention_curves_stacked(
    fault_data:     list[dict],
    dataset_filter: str | None = None,
    output:         str        = "results/figures/fault_retention_curves.png",
) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    agg = aggregate_retention(fault_data, dataset_filter)

    present_faults = [f for f in FAULT_ORDER if f in agg]
    if not present_faults:
        print("No fault data to plot.")
        return

    n_cols = 2
    n_rows = (len(present_faults) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    codings_present = sorted(
        {c for fd in agg.values() for c in fd},
        key=lambda c: SCHEME_ORDER.index(c) if c in SCHEME_ORDER else 99,
    )

    for ax_idx, fault_name in enumerate(present_faults):
        ax       = axes_flat[ax_idx]
        coding_d = agg[fault_name]

        for coding in codings_present:
            if coding not in coding_d:
                continue
            int_data    = coding_d[coding]
            intensities = sorted(int_data.keys())
            means  = [int_data[x][0] for x in intensities]
            stds   = [int_data[x][1] for x in intensities]
            color  = SCHEME_COLORS.get(coding, "#888888")
            label  = SCHEME_LABELS.get(coding, coding)

            ax.plot(intensities, means, marker="o", color=color, label=label,
                    linewidth=1.8, markersize=4)
            ax.fill_between(
                intensities,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15, color=color,
            )

        ax.set_title(FAULT_LABELS.get(fault_name, fault_name), fontsize=11)
        ax.set_xlabel(_param_label(fault_name), fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_ylim(0, 1.05)

        if FAULT_SWEEPS.get(fault_name, {}).get("logscale", True) and intensities:
            ax.set_xscale("log")
            ax.set_xticks(intensities)
            ax.set_xticklabels([f"{x:g}" for x in intensities], fontsize=9)
            ax.minorticks_off()
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )

        if ax_idx == 0:
            ax.legend(fontsize=9, loc="lower left")

    for ax in axes_flat[len(present_faults):]:
        ax.set_visible(False)

    title_suffix = f" [{dataset_filter}]" if dataset_filter else ""
    fig.suptitle(
        f"Accuracy Retention Under Hardware Faults{title_suffix}"
        "  (mean ± s.d. across runs)",
        fontsize=15, y=1.005,
    )
    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)
    print(f"Retention curves (stacked 2x2) -> {output}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fault retention curves in a 2x2 stacked layout"
    )
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--dataset",     default=None,
                   help="Filter by dataset (mnist | cifar10)")
    p.add_argument("--category",    choices=CATEGORIES, default="transient",
                   help="Fault category to plot (default: transient)")
    p.add_argument("--output-dir",  default="results/figures")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    fault_data = load_fault_results(args.results_dir)
    if not fault_data:
        print(
            "No fault_results.json files found in results/raw/.\n"
            "Run experiments/run_fault_evaluation.py first."
        )
        return
    fault_data = select_category(fault_data, args.category)

    out_dir = Path(args.output_dir)
    datasets = [args.dataset] if args.dataset else \
        sorted({r.get("dataset") for r in fault_data if r.get("dataset")})
    for ds in datasets:
        suffix = ("" if args.category == "transient" else f"_{args.category}") + f"_{ds}"
        plot_retention_curves_stacked(
            fault_data,
            dataset_filter = ds,
            output         = str(out_dir / f"fault_retention_curves{suffix}.png"),
        )


if __name__ == "__main__":
    main()
