"""Latency analysis: time-to-decision for each coding scheme."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    load_all_runs, group_runs, rebuild_pipeline,
    set_plot_style, savefig,
    SCHEME_ORDER, SCHEME_COLORS, SCHEME_LABELS,
    RULE_ORDER,   RULE_COLORS,   RULE_LABELS,
)


def _cumulative_accuracy(model, loader, device, max_batches: int = 20):
    """Accuracy at each time step t using cumulative logit sum (steps 0..t)."""
    import torch

    all_cumulative = []
    all_labels     = []
    n_seen         = 0

    model.eval()
    with torch.no_grad():
        for i, (spikes, labels) in enumerate(loader):
            if i >= max_batches:
                break
            spikes = spikes.to(device).permute(1, 0, 2).contiguous()
            labels = labels.to(device).long()

            spk_rec, _ = model(spikes)
            cumulative = spk_rec.cumsum(dim=0)

            all_cumulative.append(cumulative.cpu())
            all_labels.append(labels.cpu())
            n_seen += labels.shape[0]

    if not all_cumulative:
        return []

    cum_stack = torch.cat(
        [c.permute(1, 0, 2) for c in all_cumulative], dim=0
    )
    labels_all = torch.cat(all_labels, dim=0)

    T = cum_stack.shape[1]
    accs = []
    for t in range(T):
        preds = cum_stack[:, t, :].argmax(dim=1)
        accs.append((preds == labels_all).float().mean().item())
    return accs


def _time_to_stable(model, loader, device, max_batches: int = 10):
    """Per-sample time step at which the cumulative argmax first locks in on the final (T-step) decision and does not change again."""
    import torch

    ttd = []
    model.eval()

    with torch.no_grad():
        for i, (spikes, labels) in enumerate(loader):
            if i >= max_batches:
                break
            spikes = spikes.to(device).permute(1, 0, 2).contiguous()
            labels = labels.to(device).long()
            spk_rec, _ = model(spikes)
            cumulative  = spk_rec.cumsum(dim=0)

            T, B, C = cumulative.shape
            final_pred = cumulative[-1].argmax(dim=1)

            for b in range(B):
                if final_pred[b].item() != labels[b].item():
                    continue

                preds_b = cumulative[:, b, :].argmax(dim=1)
                fp      = final_pred[b].item()
                t_star  = T - 1
                for t in range(T - 1, -1, -1):
                    if preds_b[t].item() == fp:
                        t_star = t
                    else:
                        break
                ttd.append(t_star)

    return ttd


def plot_latency_analysis(
    results_dir:    str        = "results/raw",
    group_by:       str        = "coding",
    dataset_filter: str | None = None,
    max_batches:    int        = 20,
    figsize:        tuple      = (13, 5),
    output:         str        = "results/figures/latency_analysis.png",
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    set_plot_style()

    runs = load_all_runs(results_dir)
    if dataset_filter:
        runs = [r for r in runs if r.get("dataset") == dataset_filter]
    if not runs:
        print("No runs found.")
        return

    if group_by == "coding":
        order     = [g for g in SCHEME_ORDER] + []
        color_map = SCHEME_COLORS
        label_map = SCHEME_LABELS
    else:
        order     = RULE_ORDER[:]
        color_map = RULE_COLORS
        label_map = RULE_LABELS

    groups = group_runs(runs, by=group_by)
    order  = [g for g in order if g in groups] + \
             [g for g in groups if g not in order]

    cum_acc_by_group: dict[str, list[list[float]]] = {}
    ttd_by_group:     dict[str, list[float]]        = {}

    for gname in order:
        group_runs_ = groups[gname]
        cum_acc_by_group[gname] = []
        ttd_all: list[int] = []

        for run in group_runs_:
            run_dir = run["run_dir"]
            if run.get("config", {}).get("synthetic", False):
                continue
            try:
                stored = run.get("activity") or {}
                if stored.get("acc_by_timestep") and stored.get("decision_step_hist"):
                    cum_accs = stored["acc_by_timestep"]
                    ttd      = [t for t, c in enumerate(stored["decision_step_hist"])
                                for _ in range(c)]
                else:
                    model, loader, device, _ = rebuild_pipeline(
                        run_dir, batch_size=256
                    )
                    cum_accs = _cumulative_accuracy(model, loader, device, max_batches)
                    ttd      = _time_to_stable(model, loader, device, max_batches)
                if cum_accs:
                    cum_acc_by_group[gname].append(cum_accs)
                ttd_all.extend(ttd)
            except Exception as e:
                print(f"  Skipped {run['run_name']}: {e}")

        if ttd_all:
            ttd_by_group[gname] = ttd_all

    if not any(cum_acc_by_group.values()):
        print("No temporal data collected — are there non-synthetic completed runs?")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for gname in order:
        traces = cum_acc_by_group.get(gname)
        if not traces:
            continue
        T      = len(traces[0])
        t_axis = np.arange(1, T + 1)
        arr    = np.array(traces) * 100.0
        mean_  = arr.mean(axis=0)
        std_   = arr.std(axis=0)

        color = color_map.get(gname, "#888888")
        label = label_map.get(gname, gname)
        ax1.plot(t_axis, mean_, color=color, label=label)
        ax1.fill_between(t_axis, mean_ - std_, mean_ + std_,
                         color=color, alpha=0.15)

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Cumulative accuracy vs time step\n(early-exit potential)")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax1.legend(fontsize=9)

    valid_groups = [g for g in order if g in ttd_by_group]
    if valid_groups:
        bar_means = [np.mean(ttd_by_group[g]) for g in valid_groups]
        bar_stds  = [np.std(ttd_by_group[g])  for g in valid_groups]
        bar_labels = [label_map.get(g, g)     for g in valid_groups]
        colors     = [color_map.get(g, "#888") for g in valid_groups]
        x_pos      = np.arange(len(valid_groups))

        bars = ax2.bar(x_pos, bar_means, yerr=bar_stds,
                       color=colors, alpha=0.85,
                       capsize=4, error_kw={"elinewidth": 1.2})
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bar_labels, rotation=25, ha="right")
        ax2.set_ylabel("Time step (0-indexed)")
        ax2.set_title("Mean time-to-stable-decision\n(correct predictions only)")
        ax2.set_ylim(bottom=0)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        for bar, mean_ in zip(bars, bar_means):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(bar_stds) * 0.1,
                     f"{mean_:.1f}", ha="center", va="bottom", fontsize=9)

    title = f"Latency Analysis  ·  grouped by {group_by}"
    if dataset_filter:
        title += f"  [{dataset_filter}]"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    savefig(fig, output)
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description="Latency / time-to-decision analysis")
    p.add_argument("--results-dir",  default="results/raw")
    p.add_argument("--group-by",     default="coding",
                   choices=["coding", "learning", "dataset"])
    p.add_argument("--dataset",      default=None)
    p.add_argument("--max-batches",  type=int, default=20)
    p.add_argument("--output",       default="results/figures/latency_analysis.png")
    return p.parse_args()


def main():
    args = _parse_args()
    plot_latency_analysis(
        results_dir    = args.results_dir,
        group_by       = args.group_by,
        dataset_filter = args.dataset,
        max_batches    = args.max_batches,
        output         = args.output,
    )


if __name__ == "__main__":
    main()
