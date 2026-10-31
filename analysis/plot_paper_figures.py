"""Summary benchmark figures: fault curves, latency, accuracy-robustness trade-off, fault scale, sparsity."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import savefig
from analysis.fault_statistics import load_losses

RAW = Path("results/raw")
OUT = Path("results/figures")

CODING = [("rate", "Rate"), ("population", "Population"), ("ttfs", "TTFS"),
          ("burst", "Burst"), ("phase", "Phase")]
LEARNING = [("surrogate_gradient", "Surrogate grad."), ("bptt", "BPTT"),
            ("eprop", "e-prop"), ("stdp", "STDP")]
DATASETS = [("mnist", "MNIST"), ("cifar10", "CIFAR-10")]
LRN_COLOUR = {"surrogate_gradient": "#2a78d6", "bptt": "#008300",
              "eprop": "#e87ba4", "stdp": "#eda100"}
LRN_MARKER = {"surrogate_gradient": "o", "bptt": "s", "eprop": "D", "stdp": "^"}
COD_COLOUR = {"rate": "#2a78d6", "population": "#1baf7a", "ttfs": "#e34948",
              "burst": "#eda100", "phase": "#008300"}
COD_MARKER = {"rate": "o", "population": "s", "ttfs": "D", "burst": "^", "phase": "v"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
WEIGHT_FAULTS = [
    ("Bit flips, 4-bit digital IMC", ("digital_imc",), "#2a78d6", "o"),
    ("Bit flips, 8-bit digital neuro.", ("digital_neuromorphic",), "#1baf7a", "s"),
    ("Stuck memory blocks", ("digital_imc_block", "digital_neuromorphic_block"), "#eda100", "^"),
    ("Cells stuck ON", ("analog_imc_stuck_on", "analog_neuromorphic_stuck_on"), "#e34948", "D"),
    ("Cells stuck OFF", ("analog_imc_stuck_off", "analog_neuromorphic_stuck_off"), "#008300", "v"),
]
FAULT_TYPES = [
    ("Weight bit flips\n(digital)", ("digital_imc", "digital_neuromorphic"),
     "Bit-error rate"),
    ("Membrane upsets\n(digital)",
     ("digital_imc_membrane", "digital_neuromorphic_membrane"), "Upsets per bit and step"),
    ("Stuck memory blocks\n(digital)",
     ("digital_imc_block", "digital_neuromorphic_block"), "Fraction of weight bits"),
    ("Cells stuck ON\n(analogue)",
     ("analog_imc_stuck_on", "analog_neuromorphic_stuck_on"), "Fraction of cells"),
    ("Cells stuck OFF\n(analogue)",
     ("analog_imc_stuck_off", "analog_neuromorphic_stuck_off"), "Fraction of cells"),
]
COL_W = 3.5
PAGE_W = 7.16


def style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5, "axes.labelsize": 7.5,
        "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.4, "pdf.fonttype": 42,
    })


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, pad=3)


def runs() -> list[dict]:
    out = []
    for f in sorted(RAW.glob("*/test_metrics.json")):
        name = f.parent.name
        ds, coding = name.split("_")[:2]
        learning = name.split("_", 2)[2].rsplit("_seed", 1)[0]
        a = json.loads(f.read_text())["activity"]
        hist = a["decision_step_hist"]
        out.append({"run": name, "dataset": ds, "coding": coding, "learning": learning,
                    "acc": json.loads(f.read_text())["test_acc"],
                    "synops": a["syn_ops_per_sample"], "spike_rate": a["mean_spike_rate"],
                    "acc_t": a["acc_by_timestep"],
                    "decision": sum((t + 1) * h for t, h in enumerate(hist)) / sum(hist)})
    return out


def losses() -> dict[tuple[str, str], float]:
    """(run, sweep) -> mean normalised loss over the sweep's rates."""
    recs, _ = load_losses(str(RAW))
    return {(r["run"], r["sweep"]): r["mean"] for r in recs}


def tradeoff(rs: list[dict], loss: dict) -> None:
    import matplotlib.pyplot as plt

    from scipy.stats import rankdata

    sweeps = sorted({s for _, s in loss})
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.15), sharey=True)
    for ax, (ds, ds_label) in zip(axes, DATASETS):
        sel = [r for r in rs if r["dataset"] == ds]
        pct = defaultdict(list)
        for s in sweeps:
            ranks = rankdata([loss[(r["run"], s)] for r in sel])
            for r, k in zip(sel, ranks):
                pct[r["run"]].append(100 * (k - 1) / (len(sel) - 1))
        rel = {run: mean(v) for run, v in pct.items()}
        cfg = defaultdict(list)
        for r in sel:
            cfg[(r["coding"], r["learning"])].append((100 * r["acc"], rel[r["run"]]))
        for lrn, lrn_label in LEARNING:
            pts = [(mean(p[0] for p in v), mean(p[1] for p in v))
                   for (c, l), v in cfg.items() if l == lrn]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, c=LRN_COLOUR[lrn],
                       marker=LRN_MARKER[lrn], edgecolors="white", linewidths=0.6,
                       label=lrn_label, zorder=3)
        panel_label(ax, ds_label)
        ax.set_xlabel("Fault-free accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Fault-damage rank, percentile\n(0 = least damaged)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, handletextpad=0.1,
               columnspacing=0.9, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.08, 1, 1))
    savefig(fig, OUT / "paper_tradeoff", dpi=300)
    plt.close(fig)


def latency(rs: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 2.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    sel = [r for r in rs if r["dataset"] == "cifar10"]
    T = len(sel[0]["acc_t"])
    steps = list(range(1, T + 1))
    for cod, label in CODING:
        curves = [r["acc_t"] for r in sel if r["coding"] == cod]
        avg = [100 * mean(c[t] for c in curves) for t in range(T)]
        a.plot(steps, avg, color=COD_COLOUR[cod], marker=COD_MARKER[cod], markersize=3.2,
               markevery=[0, 4, 9, 14, 19], markeredgecolor="white", markeredgewidth=0.4,
               label=label)
    a.set_xlim(1, T)
    a.set_xticks([1, 5, 10, 15, 20])
    a.set_xlabel("Inference stopped after step")
    a.set_ylabel("CIFAR-10 accuracy (%)")
    panel_label(a, "(a)")
    a.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.1, labelspacing=0.2)

    for ds, ds_label in DATASETS:
        for cod, _ in CODING:
            grp = [r for r in rs if r["dataset"] == ds and r["coding"] == cod]
            x = mean(r["decision"] for r in grp)
            y = mean(r["synops"] for r in grp) / 1e6
            filled = ds == "cifar10"
            b.scatter([x], [y], s=20, marker=COD_MARKER[cod], linewidths=0.9, zorder=3,
                      c=COD_COLOUR[cod] if filled else "white",
                      edgecolors="white" if filled else COD_COLOUR[cod])
    b.set_yscale("log")
    b.set_xticks([1, 3, 5, 7, 9])
    b.set_xlim(0.5, 9.5)
    b.set_xlabel("Mean decision step")
    b.set_ylabel("SynOps per inference (M)")
    panel_label(b, "(b)")
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=4,
                             label="CIFAR-10"),
                      Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                             markeredgecolor=INK2, markersize=4, label="MNIST")],
             loc="lower right", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    savefig(fig, OUT / "paper_latency", dpi=300)
    plt.close(fig)


def fault_scale() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    by_rate = defaultdict(lambda: defaultdict(list))
    for f in RAW.glob("*/fault_results.json"):
        chance = 1.0 / json.loads((f.parent / "config.json").read_text()).get("n_classes", 10)
        sweeps = json.loads(f.read_text())["fault_sweeps"]
        for label, names, _, _ in WEIGHT_FAULTS:
            for n in names:
                sw = sweeps[n]
                room = sw["substrate_acc"] - chance
                for r in sw["results"]:
                    by_rate[label][r["intensity"]].append((sw["substrate_acc"] - r["mean_acc"]) / room)

    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    for label, _, colour, marker in WEIGHT_FAULTS:
        rates = sorted(by_rate[label])
        vals = [by_rate[label][k] for k in rates]
        m = [100 * mean(v) for v in vals]
        lo = [100 * np.percentile(v, 25) for v in vals]
        hi = [100 * np.percentile(v, 75) for v in vals]
        ax.fill_between(rates, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(rates, m, color=colour, marker=marker, markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.4, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Fraction of faulty storage elements (bits or cells)")
    ax.set_ylabel("Above-chance accuracy lost (%)")
    ax.axvline(0.01, color=AXIS, linewidth=0.6, zorder=1)
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    savefig(fig, OUT / "paper_fault_scale", dpi=300)
    plt.close(fig)


def fault_curves() -> None:
    import matplotlib.pyplot as plt

    pts = defaultdict(list)
    for f in RAW.glob("*/fault_results.json"):
        d = json.loads(f.read_text())
        chance = 1.0 / json.loads((f.parent / "config.json").read_text()).get("n_classes", 10)
        for name, sw in d["fault_sweeps"].items():
            room = sw["substrate_acc"] - chance
            for r in sw["results"]:
                l = 100 * (sw["substrate_acc"] - r["mean_acc"]) / room
                pts[("learning", d["learning"], d["dataset"], name, r["intensity"])].append(l)
                pts[("coding", d["coding"], d["dataset"], name, r["intensity"])].append(l)

    rows = [("learning", LEARNING, ("mnist", "cifar10"), "Learning rules\n(both datasets)",
             LRN_COLOUR, LRN_MARKER),
            ("coding", CODING, ("mnist",), "Coding schemes\n(MNIST)", COD_COLOUR, COD_MARKER),
            ("coding", CODING, ("cifar10",), "Coding schemes\n(CIFAR-10)", COD_COLOUR, COD_MARKER)]
    fig, axes = plt.subplots(len(rows), len(FAULT_TYPES), figsize=(PAGE_W, 5.0),
                             sharex="col", sharey="col")
    for i, (factor, groups, datasets, row_label, colours, markers) in enumerate(rows):
        for j, (title, sweeps, xlabel) in enumerate(FAULT_TYPES):
            ax = axes[i, j]
            rates = sorted({k[4] for k in pts if k[3] == sweeps[0]})
            for v, label in groups:
                y = [mean(x for ds in datasets for s in sweeps
                          for x in pts[(factor, v, ds, s, k)]) for k in rates]
                ax.plot(rates, y, color=colours[v], marker=markers[v], markersize=2.8,
                        markeredgecolor="white", markeredgewidth=0.3, linewidth=1.1,
                        label=label)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(title, color=INK, fontsize=7, pad=3)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(row_label + "\nabove-chance acc. lost (%)")
                ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.1,
                          labelspacing=0.15, handlelength=1.4, fontsize=6)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.4)
    savefig(fig, OUT / "paper_fault_curves", dpi=300)
    plt.close(fig)


def sparsity(rs: list[dict], loss: dict) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats

    membrane = ("digital_imc_membrane", "digital_neuromorphic_membrane")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.15))
    for ax, (ds, ds_label) in zip(axes, DATASETS):
        sel = [r for r in rs if r["dataset"] == ds]
        y = {r["run"]: 100 * mean(loss[(r["run"], s)] for s in membrane) for r in sel}
        for cod, label in CODING:
            grp = [r for r in sel if r["coding"] == cod]
            ax.scatter([100 * r["spike_rate"] for r in grp], [y[r["run"]] for r in grp],
                       s=12, c=COD_COLOUR[cod], marker=COD_MARKER[cod], edgecolors="white",
                       linewidths=0.5, label=label, zorder=3)
        rho = stats.spearmanr([r["spike_rate"] for r in sel], [y[r["run"]] for r in sel])[0]
        panel_label(ax, f"{ds_label}, Spearman $\\rho = {rho:.2f}$")
    axes[0].set_ylabel("Above-chance accuracy\nlost to membrane upsets (%)")
    fig.supxlabel("Spike rate (% of neuron steps)", fontsize=7.5, y=0.09)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, handletextpad=0.05,
               columnspacing=0.8, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.12, 1, 1))
    savefig(fig, OUT / "paper_sparsity", dpi=300)
    plt.close(fig)


def main() -> None:
    style()
    rs, loss = runs(), losses()
    tradeoff(rs, loss)
    latency(rs)
    fault_scale()
    fault_curves()
    sparsity(rs, loss)


if __name__ == "__main__":
    main()
