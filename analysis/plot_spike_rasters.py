"""Visualise the spike train that enters the first SNN layer for a few test samples."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import set_plot_style, savefig, rebuild_pipeline


def plot_spike_rasters(
    run_name:    str,
    results_dir: str   = "results/raw",
    n_samples:   int   = 4,
    n_neurons:   int   = 150,
    rng_seed:    int   = 0,
    figsize:     tuple = (12, 8),
    output:      str   | None = None,
) -> None:
    """Plot the input spike rasters of a few test samples of one run."""
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    set_plot_style()

    run_dir = Path(results_dir) / run_name
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    print(f"Loading pipeline from {run_dir} ...")
    model, test_loader, device, config = rebuild_pipeline(str(run_dir))

    batch_spikes, batch_labels = next(iter(test_loader))
    batch_spikes = batch_spikes[:n_samples]
    batch_labels = batch_labels[:n_samples]

    S, T, N = batch_spikes.shape

    rng          = np.random.default_rng(rng_seed)
    neuron_idx   = rng.choice(N, size=min(n_neurons, N), replace=False)
    neuron_idx   = np.sort(neuron_idx)

    coding_name = config.get("coding",  {})
    if isinstance(coding_name, dict):
        coding_name = coding_name.get("name", "")
    ds_name = config.get("dataset", {})
    if isinstance(ds_name, dict):
        ds_name = ds_name.get("name", "")

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(S, 1, hspace=0.45)

    for s in range(S):
        ax    = fig.add_subplot(gs[s])
        spk_s = batch_spikes[s].numpy()
        spk_sub = spk_s[:, neuron_idx]

        t_idx, n_idx = np.where(spk_sub > 0.5)
        ax.scatter(t_idx, n_idx, s=1.5, c="steelblue", alpha=0.7, rasterized=True)

        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.5, len(neuron_idx) - 0.5)
        ax.set_ylabel("Neuron", fontsize=9)
        label = int(batch_labels[s].item())

        rate = float(spk_s.mean())
        ax.set_title(f"Sample {s+1}  (class={label})  "
                     f"mean rate={rate:.3f}", fontsize=10)

        if s == S - 1:
            ax.set_xlabel("Time step")
        else:
            ax.set_xticklabels([])

    fig.suptitle(
        f"Input Spike Raster — {ds_name.upper()}  ·  {coding_name} encoding\n"
        f"({len(neuron_idx)} of {N} neurons shown)",
        fontsize=12, y=1.01,
    )

    if output is None:
        output = f"results/figures/raster_{run_name}.png"
    savefig(fig, output)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spike raster for a run's input encoding")
    p.add_argument("--run-name",     required=True)
    p.add_argument("--results-dir",  default="results/raw")
    p.add_argument("--n-samples",    type=int, default=4)
    p.add_argument("--n-neurons",    type=int, default=150)
    p.add_argument("--output",       default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_spike_rasters(
        run_name    = args.run_name,
        results_dir = args.results_dir,
        n_samples   = args.n_samples,
        n_neurons   = args.n_neurons,
        output      = args.output,
    )


if __name__ == "__main__":
    main()
