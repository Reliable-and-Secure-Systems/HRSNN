"""Synaptic-operation (SynOps) energy-efficiency analysis across runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    load_all_runs, group_runs,
    set_plot_style, savefig,
    SCHEME_COLORS, SCHEME_LABELS, SCHEME_ORDER,
    RULE_COLORS, RULE_LABELS,
    rebuild_pipeline,
)


def measure_synops(
    model,
    test_loader,
    device,
    max_batches: int | None = 10,
) -> dict:
    """Count spikes and SynOps by re-running inference with forward hooks."""
    from src.tracking.activity import measure_activity
    return measure_activity(model, test_loader, device, max_batches)


def run_activity(run: dict, max_batches: int | None, batch_size: int = 128) -> dict:
    """Return a run's stored activity block, or measure it from the checkpoint if missing."""
    stored = run.get("activity")
    if stored:
        return stored
    print(f"  {run['run_name']}: no stored activity, measuring from checkpoint "
          f"(experiments/record_activity.py stores it)")
    model, loader, device, _ = rebuild_pipeline(run["run_dir"], batch_size=batch_size)
    return measure_synops(model, loader, device, max_batches)


def plot_energy_efficiency(
    results_dir:    str        = "results/raw",
    group_by:       str        = "coding",
    dataset_filter: str | None = None,
    max_batches:    int        = 20,
    figsize:        tuple      = (13, 5),
    output:         str        = "results/figures/energy_efficiency.png",
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    set_plot_style()

    runs = load_all_runs(results_dir)
    if dataset_filter:
        runs = [r for r in runs if r.get("dataset") == dataset_filter]
    if not runs:
        print("No runs found.")
        return

    energy_data = []
    for run in runs:
        if run.get("config", {}).get("synthetic", False):
            continue
        try:
            stats = run_activity(run, max_batches)
            energy_data.append({**run, **stats})
        except Exception as e:
            print(f"  Skipped {run['run_name']}: {e}")

    if not energy_data:
        print("No energy data collected.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    grouped = group_runs(energy_data, by=group_by)
    if group_by == "coding":
        order = [g for g in SCHEME_ORDER if g in grouped] + \
                [g for g in grouped if g not in SCHEME_ORDER]
        color_map = SCHEME_COLORS
        label_map = SCHEME_LABELS
    else:
        order     = sorted(grouped.keys())
        color_map = RULE_COLORS
        label_map = RULE_LABELS

    x_pos  = range(len(order))
    width  = 0.35
    ann_ref = None

    for i, gname in enumerate(order):
        group    = grouped[gname]
        syn_ops  = np.mean([r["syn_ops_per_sample"] for r in group])
        ann_macs = np.mean([r["ann_macs"]            for r in group])

        color = color_map.get(gname, "#888888")
        label = label_map.get(gname, gname)

        ax1.bar(i - width/2, syn_ops,  width, color=color, alpha=0.85, label=label)
        if ann_ref is None:
            ann_ref = ann_macs

    if ann_ref:
        ax1.axhline(ann_ref, color="red", linestyle="--", linewidth=1.2, label="ANN MACs")

    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels(
        [label_map.get(g, g) for g in order],
        rotation=25, ha="right",
    )
    ax1.set_ylabel("Operations per sample")
    ax1.set_title("SynOps vs ANN MACs")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_yscale("log")

    for r in energy_data:
        gname  = r.get(group_by, "?")
        color  = color_map.get(gname, "#888888")
        label  = label_map.get(gname, gname)
        rate   = r.get("mean_spike_rate",    0)
        acc    = r.get("test_acc",           0)
        ax2.scatter(rate, acc, color=color, s=60, alpha=0.8, label=label, zorder=3)
        ax2.annotate(
            label, (rate, acc),
            textcoords="offset points", xytext=(5, 2), fontsize=7,
        )

    ax2.set_xlabel("Mean spike rate (firing probability)")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("Efficiency frontier: spike rate vs accuracy")
    ax2.set_xlim(left=0)

    handles, labels_ = ax2.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen[l] = h
    ax2.legend(seen.values(), seen.keys(), fontsize=8)

    fig.suptitle(
        f"Energy Efficiency  ·  grouped by {group_by}"
        + (f"  [{dataset_filter}]" if dataset_filter else ""),
        fontsize=12,
    )
    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SynOps energy efficiency analysis")
    p.add_argument("--results-dir",  default="results/raw")
    p.add_argument("--group-by",     default="coding",
                   choices=["coding", "learning", "dataset"])
    p.add_argument("--dataset",      default=None)
    p.add_argument("--max-batches",  type=int, default=20,
                   help="Batches per run used for counting (more = slower but accurate)")
    p.add_argument("--output",       default="results/figures/energy_efficiency.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_energy_efficiency(
        results_dir    = args.results_dir,
        group_by       = args.group_by,
        dataset_filter = args.dataset,
        max_batches    = args.max_batches,
        output         = args.output,
    )


if __name__ == "__main__":
    main()
