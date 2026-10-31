"""Power and energy estimation across coding schemes and hardware platforms."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    load_all_runs, group_runs,
    set_plot_style, savefig,
    SCHEME_ORDER, SCHEME_COLORS, SCHEME_LABELS,
    RULE_ORDER,   RULE_COLORS,   RULE_LABELS,
)
from analysis.plot_energy_efficiency import run_activity


HARDWARE_ENERGY: dict[str, tuple[float, float, str]] = {
    "digital_imc":  (0.5,   2.0,  "Digital IMC (CMOS)"),
    "analog_imc":   (0.05,  4.6,  "Analog IMC (RRAM)"),
    "digital_neuro": (3.5,  7.0,  "Digital Neuro. (Loihi)"),
    "analog_neuro":  (0.05, 4.6,  "Analog Neuro. (RRAM)"),
}


def compute_energy(
    syn_ops_per_sample: float,
    ann_macs:           float,
    e_ac_pj:            float,
    e_mac_pj:           float,
) -> dict:
    """Convert operation counts to energy in picojoules."""
    e_snn = syn_ops_per_sample * e_ac_pj
    e_ann = ann_macs           * e_mac_pj
    ratio = e_snn / max(e_ann, 1e-12)
    return {
        "e_snn_pj":        e_snn,
        "e_ann_pj":        e_ann,
        "efficiency_ratio": ratio,
        "savings_pct":     max(0.0, (1.0 - ratio) * 100.0),
    }


def plot_power_estimation(
    results_dir:    str        = "results/raw",
    group_by:       str        = "coding",
    dataset_filter: str | None = None,
    hardware_names: list[str]  | None = None,
    max_batches:    int        = 10,
    figsize:        tuple      = (15, 5),
    output:         str        = "results/figures/power_estimation.png",
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    set_plot_style()

    hw_names = hardware_names or list(HARDWARE_ENERGY.keys())
    hw_items = [(n, HARDWARE_ENERGY[n]) for n in hw_names if n in HARDWARE_ENERGY]

    runs = load_all_runs(results_dir)
    if dataset_filter:
        runs = [r for r in runs if r.get("dataset") == dataset_filter]
    if not runs:
        print("No runs found.")
        return

    if group_by == "coding":
        order     = SCHEME_ORDER[:]
        color_map = SCHEME_COLORS
        label_map = SCHEME_LABELS
    else:
        order     = RULE_ORDER[:]
        color_map = RULE_COLORS
        label_map = RULE_LABELS

    groups = group_runs(runs, by=group_by)
    order  = [g for g in order if g in groups] + \
             [g for g in groups if g not in order]

    print("Measuring synaptic operations per run...")
    synop_data: list[dict] = []

    for run in runs:
        if run.get("config", {}).get("synthetic", False):
            continue
        try:
            stats = run_activity(run, max_batches, batch_size=256)
            synop_data.append({**run, **stats})
        except Exception as e:
            print(f"  Skipped {run['run_name']}: {e}")

    if not synop_data:
        print("No SynOps data collected.")
        return

    grouped = group_runs(synop_data, by=group_by)

    agg: dict[str, dict] = {}
    for gname in order:
        if gname not in grouped:
            continue
        g = grouped[gname]
        agg[gname] = {
            "syn_ops":  np.mean([r["syn_ops_per_sample"] for r in g]),
            "ann_macs": np.mean([r["ann_macs"]           for r in g]),
            "test_acc": np.mean([r.get("test_acc", 0)    for r in g]) * 100,
            "runs":     g,
        }

    valid_groups = [g for g in order if g in agg]
    if not valid_groups:
        print("No data after aggregation.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    n_groups  = len(valid_groups)
    n_hw      = len(hw_items)
    width     = 0.8 / n_hw
    x_pos     = np.arange(n_groups)

    for hi, (hw_name, (e_ac, e_mac, hw_label)) in enumerate(hw_items):
        energies = []
        for gname in valid_groups:
            e = compute_energy(agg[gname]["syn_ops"], agg[gname]["ann_macs"],
                               e_ac, e_mac)
            energies.append(e["e_snn_pj"])

        offset = (hi - n_hw / 2 + 0.5) * width
        ax1.bar(x_pos + offset, energies, width * 0.9,
                label=hw_label, alpha=0.85)

    e_ac_ref, e_mac_ref, _ = HARDWARE_ENERGY["45nm_digital"]
    ann_ref = np.mean([
        agg[g]["ann_macs"] * e_mac_ref for g in valid_groups
    ])
    ax1.axhline(ann_ref, color="red", linestyle="--", linewidth=1.2,
                label=f"ANN ref (45nm, {ann_ref:.0f} pJ)")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([label_map.get(g, g) for g in valid_groups],
                        rotation=25, ha="right")
    ax1.set_ylabel("Energy per inference (pJ)")
    ax1.set_title("Absolute energy\nvs ANN baseline")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)

    for hi, (hw_name, (e_ac, e_mac, hw_label)) in enumerate(hw_items):
        ratios = []
        for gname in valid_groups:
            e = compute_energy(agg[gname]["syn_ops"], agg[gname]["ann_macs"],
                               e_ac, e_mac)
            ratios.append(e["efficiency_ratio"] * 100)

        offset = (hi - n_hw / 2 + 0.5) * width
        ax2.bar(x_pos + offset, ratios, width * 0.9,
                label=hw_label, alpha=0.85)

    ax2.axhline(100, color="red", linestyle="--", linewidth=1.2,
                label="ANN = 100%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([label_map.get(g, g) for g in valid_groups],
                        rotation=25, ha="right")
    ax2.set_ylabel("E_SNN / E_ANN  (%)")
    ax2.set_title("Energy ratio vs ANN\n(lower = more efficient)")
    ax2.legend(fontsize=8)

    e_ac_default, e_mac_default, hw_lbl = HARDWARE_ENERGY["45nm_digital"]

    for run in synop_data:
        gname  = run.get(group_by, "?")
        color  = color_map.get(gname, "#888888")
        label  = label_map.get(gname, gname)
        e      = compute_energy(run["syn_ops_per_sample"], run["ann_macs"],
                                e_ac_default, e_mac_default)
        acc    = run.get("test_acc", 0) * 100
        ax3.scatter(e["e_snn_pj"], acc,
                    color=color, s=60, alpha=0.8, label=label, zorder=3)

    ann_e_mean = np.mean([
        run["ann_macs"] * e_mac_default for run in synop_data
    ])
    ax3.axvline(ann_e_mean, color="red", linestyle="--", linewidth=1.2,
                label=f"ANN ({hw_lbl})")

    ax3.set_xlabel(f"Energy per inference (pJ)  [{hw_lbl}]")
    ax3.set_ylabel("Test accuracy (%)")
    ax3.set_title("Pareto frontier\n(top-left = best trade-off)")
    ax3.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y:.0f}%")
    )

    handles, labels_ = ax3.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen[l] = h
    ax3.legend(seen.values(), seen.keys(), fontsize=8)

    title = f"Power Estimation  ·  grouped by {group_by}"
    if dataset_filter:
        title += f"  [{dataset_filter}]"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)

    print(f"\n{'='*70}")
    print(f"Energy summary  |  group_by={group_by}")
    print(f"{'='*70}")
    col_w = 12
    header = f"{'Group':<22}"
    for _, (_, _, hw_lbl) in hw_items:
        header += f"  {hw_lbl[:col_w]:>{col_w}}"
    print(header)
    print("-" * len(header))
    for gname in valid_groups:
        row = f"{label_map.get(gname, gname):<22}"
        for _, (e_ac, e_mac, _) in hw_items:
            e = compute_energy(agg[gname]["syn_ops"], agg[gname]["ann_macs"],
                               e_ac, e_mac)
            row += f"  {e['e_snn_pj']:>{col_w}.2f}"
        print(row)
    print(f"\n{'ANN baseline (45nm)':<22}  "
          f"{agg[valid_groups[0]]['ann_macs'] * e_mac_ref:{col_w}.2f} pJ")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Energy / power estimation across hardware platforms"
    )
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--group-by",    default="coding",
                   choices=["coding", "learning", "dataset"])
    p.add_argument("--dataset",     default=None)
    p.add_argument("--hardware",    nargs="+",
                   default=list(HARDWARE_ENERGY.keys()),
                   choices=list(HARDWARE_ENERGY.keys()),
                   help="Hardware platforms to include")
    p.add_argument("--max-batches", type=int, default=10)
    p.add_argument("--output",      default="results/figures/power_estimation.png")
    return p.parse_args()


def main():
    args = _parse_args()
    plot_power_estimation(
        results_dir    = args.results_dir,
        group_by       = args.group_by,
        dataset_filter = args.dataset,
        hardware_names = args.hardware,
        max_batches    = args.max_batches,
        output         = args.output,
    )


if __name__ == "__main__":
    main()
