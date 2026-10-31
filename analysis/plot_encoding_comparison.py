"""Show the same image encoded with all 5 coding schemes side by side."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import set_plot_style, savefig, SCHEME_ORDER, SCHEME_LABELS


def plot_encoding_comparison(
    dataset:    str   = "mnist",
    sample_idx: int   = 0,
    time_steps: int   = 20,
    data_dir:   str   = "data/",
    figsize:    tuple = (14, 13),
    output:     str   = "results/figures/encoding_comparison.png",
) -> None:
    """Plot one test image under every coding scheme side by side."""
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import torchvision
    import torchvision.transforms as T

    from src.coding.registry import get_encoder

    set_plot_style()

    transform = T.Compose([T.ToTensor()])
    if dataset == "mnist":
        ds      = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                              transform=transform)
        H, W, C = 28, 28, 1
    elif dataset == "cifar10":
        ds      = torchvision.datasets.CIFAR10(data_dir, train=False, download=True,
                                               transform=transform)
        H, W, C = 32, 32, 3
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    img_tensor, label = ds[sample_idx]
    flat = img_tensor.view(1, -1).float()
    N    = flat.shape[1]

    scheme_params: dict[str, dict] = {
        "rate":       {},
        "ttfs":       {},
        "phase":      {},
        "burst":      {},
        "population": {"n_gaussians": 10, "sigma": 0.5},
    }

    n_schemes  = len(SCHEME_ORDER)
    n_show_n   = 10

    fig = plt.figure(figsize=figsize)
    n_rows  = n_schemes + 1
    gs      = gridspec.GridSpec(n_rows, 3, wspace=0.35, hspace=0.55,
                                 width_ratios=[1, 2.5, 3])

    ax_img  = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_ts0  = fig.add_subplot(gs[0, 2])

    img_np = img_tensor.permute(1, 2, 0).numpy()
    if C == 1:
        ax_img.imshow(img_np.squeeze(), cmap="gray", interpolation="nearest")
    else:
        ax_img.imshow(np.clip(img_np, 0, 1))
    ax_img.set_title(f"Original\n(class={label})", fontsize=10)
    ax_img.axis("off")

    ax_text.axis("off")
    ax_text.text(0.05, 0.5, f"Dataset: {dataset.upper()}\n"
                             f"Pixels: {H}×{W}×{C} = {N}\n"
                             f"T = {time_steps} steps",
                 transform=ax_text.transAxes, va="center", fontsize=10)

    ax_ts0.axis("off")

    for row_idx, scheme_name in enumerate(SCHEME_ORDER, start=1):
        params  = scheme_params.get(scheme_name, {})
        encoder = get_encoder(scheme_name, **params)

        with torch.no_grad():
            spikes = encoder(flat, time_steps)
        spikes_np = spikes.squeeze(1).numpy()

        spike_count = spikes_np.sum(axis=0)
        if spike_count.shape[0] > N:
            spike_count = spike_count[:N]

        ONE_SHOT_SCHEMES = {"ttfs", "phase"}
        if scheme_name in ONE_SHOT_SCHEMES:
            has_spike = spikes_np.max(axis=0) > 0.5
            fst = np.argmax(spikes_np, axis=0).astype(float)
            fst[~has_spike] = np.nan
            stat       = fst
            hmap_title = f"First spike time  (T={time_steps})"
            cbar_label = "Timestep"
            cmap_name  = "viridis_r"
        else:
            stat       = spike_count
            hmap_title = f"Spike count  (T={time_steps})"
            cbar_label = "Σ spikes"
            cmap_name  = "hot"

        if C == 1:
            hmap = stat.reshape(H, W)
        else:
            n_ch = stat.shape[0] // (H * W)
            n_ch = min(n_ch, C)
            hmap = stat[:n_ch * H * W].reshape(n_ch, H * W).mean(axis=0).reshape(H, W)

        ax_hmap = fig.add_subplot(gs[row_idx, 1])
        im = ax_hmap.imshow(hmap, cmap=cmap_name, interpolation="nearest",
                            aspect="equal")
        fig.colorbar(im, ax=ax_hmap, fraction=0.04, pad=0.02, label=cbar_label)
        ax_hmap.set_title(hmap_title, fontsize=8)
        ax_hmap.axis("off")

        ax_thumb = fig.add_subplot(gs[row_idx, 0])
        label_str = SCHEME_LABELS.get(scheme_name, scheme_name)
        ax_thumb.text(0.5, 0.5, label_str, ha="center", va="center",
                      fontsize=11, fontweight="bold",
                      transform=ax_thumb.transAxes)
        ax_thumb.axis("off")

        ax_ts = fig.add_subplot(gs[row_idx, 2])
        t_range = np.arange(time_steps)
        step = max(1, N // n_show_n)
        neuron_subset = np.arange(0, N, step)[:n_show_n]
        sub_spikes = spikes_np[:, neuron_subset]

        for ni in range(sub_spikes.shape[1]):
            trace = sub_spikes[:, ni]
            spike_times = t_range[trace > 0.5]
            ax_ts.scatter(spike_times, np.full_like(spike_times, ni),
                          s=8, c="steelblue", alpha=0.8)

        ax_ts.set_xlim(-0.5, time_steps - 0.5)
        ax_ts.set_ylim(-0.5, n_show_n - 0.5)
        tick_step = max(1, time_steps // 4)
        ax_ts.set_xticks(range(0, time_steps, tick_step))
        ax_ts.set_yticks([])
        ax_ts.set_xlabel("Time step", fontsize=8)
        ax_ts.set_title(f"{n_show_n} sample neurons", fontsize=8)
        mean_rate = float(spikes_np.mean())
        ax_ts.text(0.98, 0.95, f"rate={mean_rate:.3f}",
                   transform=ax_ts.transAxes, ha="right", va="top", fontsize=8)

    fig.suptitle(
        f"Encoding comparison — {dataset.upper()}  sample #{sample_idx}",
        fontsize=13, y=1.01,
    )

    savefig(fig, output)
    plt.close(fig)
    print(f"Encoding comparison saved to {output}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all 6 encoding schemes visually")
    p.add_argument("--dataset",     default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--sample-idx",  type=int, default=0,
                   help="Index of the test sample to encode")
    p.add_argument("--time-steps",  type=int, default=20)
    p.add_argument("--data-dir",    default="data/")
    p.add_argument("--output",
                   default="results/figures/encoding_comparison.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_encoding_comparison(
        dataset    = args.dataset,
        sample_idx = args.sample_idx,
        time_steps = args.time_steps,
        data_dir   = args.data_dir,
        output     = args.output,
    )


if __name__ == "__main__":
    main()
