"""Fault reliability visualization."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    set_plot_style, savefig,
    SCHEME_COLORS, SCHEME_LABELS, SCHEME_ORDER,
)


from experiments.run_fault_evaluation import (
    FAULT_SWEEPS, FAULT_MODEL_VERSION, CATEGORIES,
)

FAULT_ORDER  = list(FAULT_SWEEPS)
FAULT_LABELS = {name: sw["label"] for name, sw in FAULT_SWEEPS.items()}

PARAM_LABELS = {
    "ber":                 "Bit error rate (fraction of weight bits flipped)",
    "upsets_per_bit_step": "Upsets per membrane-register bit per time step",
    "faulty_bit_fraction": "Fraction of weight bits stuck (in blocks)",
    "stuck_cell_fraction": "Fraction of memristor cells stuck",
}

CATEGORY_TEXT = {
    "transient": {"fault": "Transient", "what": "bit flips / register upsets"},
    "permanent": {"fault": "Permanent", "what": "stuck blocks / stuck cells"},
}


def sweep_category(sweep: dict) -> str:
    """A sweep's fault category; files written before categories existed are transient."""
    return sweep.get("category", "transient")


def select_category(fault_data: list[dict], category: str) -> list[dict]:
    """Keep only the current sweeps of one fault category in each run's results."""
    out, stale = [], 0
    for run in fault_data:
        sweeps = {}
        for name, sw in run.get("fault_sweeps", {}).items():
            if sweep_category(sw) != category:
                continue
            if sw.get("fault_model_version") != FAULT_MODEL_VERSION:
                stale += 1
                continue
            sweeps[name] = sw
        out.append({**run, "fault_sweeps": sweeps})
    if stale:
        print(f"WARNING: {stale} {category} sweep(s) from an older fault model were "
              f"skipped; recompute them with experiments/run_fault_evaluation.py")
    return out


def load_fault_results(results_dir: str) -> list[dict]:
    """Return list of fault_results.json dicts from all run directories."""
    rd = Path(results_dir)
    out = []
    for run_dir in sorted(rd.iterdir()):
        fp = run_dir / "fault_results.json"
        if fp.exists():
            with open(fp) as f:
                out.append(json.load(f))
    return out


def aggregate_retention(
    fault_data: list[dict],
    dataset_filter: str | None,
) -> dict[str, dict[str, dict[float, tuple[float, float]]]]:
    """Mean and s.d. across runs of the accuracy per fault, coding scheme and intensity."""
    raw: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run in fault_data:
        if dataset_filter and run.get("dataset") != dataset_filter:
            continue
        coding = run.get("coding", "unknown")
        for fault_name, sweep in run.get("fault_sweeps", {}).items():
            for pt in sweep.get("results", []):
                raw[fault_name][coding][float(pt["intensity"])].append(
                    pt["mean_acc"]
                )

    agg: dict[str, dict[str, dict[float, tuple[float, float]]]] = {}
    for fault_name, coding_dict in raw.items():
        agg[fault_name] = {}
        for coding, int_dict in coding_dict.items():
            agg[fault_name][coding] = {
                intensity: (
                    float(np.mean(accs)),
                    float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                    len(accs),
                )
                for intensity, accs in sorted(int_dict.items())
            }

    return agg


def _mid_points(
    fault_data:     list[dict],
    dataset_filter: str | None,
    field:          str,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """{coding -> {sweep -> mean of `field` at the sweep's middle intensity}}, plus {sweep -> that intensity}."""
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    where: dict[str, float] = {}
    for run in fault_data:
        if dataset_filter and run.get("dataset") != dataset_filter:
            continue
        coding = run.get("coding", "unknown")
        for name, sweep in run.get("fault_sweeps", {}).items():
            results = sweep.get("results", [])
            if not results:
                continue
            mid = results[len(results) // 2]
            if mid.get(field) is None:
                continue
            raw[coding][name].append(mid[field])
            where[name] = float(mid["intensity"])
    means = {c: {n: float(np.mean(v)) for n, v in d.items()} for c, d in raw.items()}
    return means, where


def build_heatmap_data(fault_data, dataset_filter):
    """{coding -> {sweep -> mean accuracy at mid-sweep}}, {sweep -> intensity}."""
    return _mid_points(fault_data, dataset_filter, "mean_acc")


def build_degradation_data(fault_data, dataset_filter):
    """{coding -> {sweep -> mean degradation at mid-sweep}}, {sweep -> intensity}."""
    return _mid_points(fault_data, dataset_filter, "degradation")


def _column_label(name: str, intensity: float) -> str:
    return f"{FAULT_LABELS.get(name, name)}\n@ {intensity:g}"


def plot_retention_curves(
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

    n_cols = 4
    n_rows = (len(present_faults) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat  = np.array(axes).flatten()

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
            int_data   = coding_d[coding]
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

        ax.set_title(FAULT_LABELS.get(fault_name, fault_name), fontsize=10)
        ax.set_xlabel(_param_label(fault_name), fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(0, 1.05)

        if FAULT_SWEEPS.get(fault_name, {}).get("logscale", True) and intensities:
            ax.set_xscale("log")
            ax.set_xticks(intensities)
            ax.set_xticklabels([f"{x:g}" for x in intensities], fontsize=8)
            ax.minorticks_off()
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )

        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    for ax in axes_flat[len(present_faults):]:
        ax.set_visible(False)

    title_suffix = f" [{dataset_filter}]" if dataset_filter else ""
    fig.suptitle(
        f"Accuracy Retention Under Hardware Faults{title_suffix}"
        "  (mean ± s.d. across runs)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)
    print(f"Retention curves -> {output}")


def _param_label(fault_name: str) -> str:
    param = FAULT_SWEEPS.get(fault_name, {}).get("param", "ber")
    return PARAM_LABELS.get(param, param)


def plot_fault_heatmap(
    fault_data:     list[dict],
    dataset_filter: str | None = None,
    output:         str        = "results/figures/fault_heatmap.png",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    set_plot_style()
    heatmap, where = build_heatmap_data(fault_data, dataset_filter)

    codings  = [c for c in SCHEME_ORDER if c in heatmap]
    hardwares = [h for h in FAULT_ORDER if any(h in v for v in heatmap.values())]

    if not codings or not hardwares:
        print("Insufficient data for heatmap.")
        return

    matrix = np.full((len(codings), len(hardwares)), np.nan)
    for ci, coding in enumerate(codings):
        for hi, hw in enumerate(hardwares):
            if hw in heatmap.get(coding, {}):
                matrix[ci, hi] = heatmap[coding][hw]

    fig, ax = plt.subplots(figsize=(max(6, len(hardwares) * 1.8),
                                    max(4, len(codings) * 0.9)))

    cmap = LinearSegmentedColormap.from_list(
        "reliability", ["#d62728", "#ffbb78", "#2ca02c"]
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(hardwares)))
    ax.set_xticklabels([_column_label(h, where[h]) for h in hardwares],
                       rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(codings)))
    ax.set_yticklabels([SCHEME_LABELS.get(c, c) for c in codings])

    for ci in range(len(codings)):
        for hi in range(len(hardwares)):
            val = matrix[ci, hi]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(hi, ci, f"{val:.2%}", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")
            else:
                ax.text(hi, ci, "—", ha="center", va="center",
                        fontsize=10, color="#aaaaaa")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean accuracy under fault", fontsize=10)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    title_suffix = f" [{dataset_filter}]" if dataset_filter else ""
    ax.set_title(
        f"Fault Resilience: Coding Scheme × Fault Sweep{title_suffix}\n"
        "(mean accuracy at each sweep's middle intensity, mean over runs)",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)
    print(f"Heatmap -> {output}")


def plot_degradation_bars(
    fault_data:     list[dict],
    dataset_filter: str | None = None,
    output:         str        = "results/figures/fault_degradation_bars.png",
    category:       str        = "transient",
) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    degradation, where = build_degradation_data(fault_data, dataset_filter)

    codings   = [c for c in SCHEME_ORDER if c in degradation]
    hardwares = [h for h in FAULT_ORDER if any(h in v for v in degradation.values())]

    if not codings or not hardwares:
        print("Insufficient data for degradation bars.")
        return

    n_hw    = len(hardwares)
    n_cod   = len(codings)
    width   = 0.8 / n_cod
    x_base  = np.arange(n_hw)

    fig, ax = plt.subplots(figsize=(max(7, n_hw * 2.5), 5))

    for ci, coding in enumerate(codings):
        x_pos    = x_base + (ci - n_cod / 2 + 0.5) * width
        degs     = []

        for hw in hardwares:
            deg = degradation.get(coding, {}).get(hw, None)
            degs.append(deg if deg is not None else float("nan"))

        color  = SCHEME_COLORS.get(coding, "#888888")
        label  = SCHEME_LABELS.get(coding, coding)
        bars   = ax.bar(x_pos, degs, width * 0.9, color=color,
                        label=label, alpha=0.85)

        for bar, deg in zip(bars, degs):
            if deg > 0.01:
                ax.annotate(
                    f"{deg:.1%}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x_base)
    ax.set_xticklabels([_column_label(h, where[h]) for h in hardwares], fontsize=8)
    text = CATEGORY_TEXT[category]
    ax.set_ylabel("Degradation (substrate − faulty)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    finite = [d for d in np.ravel([[degradation.get(c, {}).get(h, np.nan)
                                    for h in hardwares] for c in codings])
              if np.isfinite(d)]
    ax.set_ylim(bottom=min(0.0, min(finite, default=0.0) * 1.2))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(fontsize=9, loc="upper right")

    title_suffix = f" [{dataset_filter}]" if dataset_filter else ""
    ax.set_title(
        f"{text['fault']} Faults: Accuracy Degradation by Coding Scheme"
        f"{title_suffix}\n(drop from each substrate's fault-free baseline, "
        f"at each sweep's middle intensity; {text['what']})",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)
    print(f"Degradation bars -> {output}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fault reliability visualization")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--dataset",     default=None,
                   help="One dataset (mnist | cifar10); default: one figure set per dataset")
    p.add_argument("--learning",    default=None,
                   help="Only runs with this learning rule (default: all, averaged)")
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
    if args.learning:
        fault_data = [r for r in fault_data if r.get("learning") == args.learning]

    out_dir = Path(args.output_dir)
    datasets = [args.dataset] if args.dataset else \
        sorted({r.get("dataset") for r in fault_data if r.get("dataset")})
    for ds in datasets:
        suffix = ("" if args.category == "transient" else f"_{args.category}") + \
                 (f"_{args.learning}" if args.learning else "") + f"_{ds}"
        plot_retention_curves(
            fault_data,
            dataset_filter = ds,
            output         = str(out_dir / f"fault_retention_curves{suffix}.png"),
        )
        plot_fault_heatmap(
            fault_data,
            dataset_filter = ds,
            output         = str(out_dir / f"fault_heatmap{suffix}.png"),
        )
        plot_degradation_bars(
            fault_data,
            dataset_filter = ds,
            output         = str(out_dir / f"fault_degradation_bars{suffix}.png"),
            category       = args.category,
        )


if __name__ == "__main__":
    main()
