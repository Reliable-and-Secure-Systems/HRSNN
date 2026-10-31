"""Per-class confusion matrix heatmap for a completed run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import (
    set_plot_style, savefig,
    rebuild_pipeline, collect_predictions,
)


def plot_confusion_matrix(
    run_name:    str,
    results_dir: str  = "results/raw",
    normalize:   bool = True,
    figsize:     tuple = (8, 7),
    output:      str  | None = None,
) -> None:
    """Plot the per-class confusion matrix of one run from its checkpoint."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    set_plot_style()

    run_dir = Path(results_dir) / run_name
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    print(f"Loading model from {run_dir} ...")
    model, test_loader, device, config = rebuild_pipeline(str(run_dir))

    print("Running inference ...")
    true_labels, pred_labels, _ = collect_predictions(model, test_loader, device)

    n_classes   = config.get("n_classes", 10)
    class_names = [str(i) for i in range(n_classes)]

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n_classes)))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = cm.astype(float) / np.maximum(row_sums, 1)
        fmt, vmin, vmax = ".2f", 0.0, 1.0
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot  = cm
        fmt, vmin, vmax = "d", None, None
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=figsize)

    try:
        import seaborn as sns
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"label": cbar_label},
                    ax=ax)
    except ImportError:
        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                       vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        thresh = (cm_plot.max() + cm_plot.min()) / 2
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_plot[i, j]
                ax.text(j, i, (f"{val:{fmt}}" if fmt != "d" else str(val)),
                        ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=8)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    ds  = config.get("dataset", {})
    ds_name = ds.get("name", "") if isinstance(ds, dict) else ""
    ax.set_title(f"Confusion Matrix — {run_name}\n(acc = {acc:.4f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.tight_layout()

    if output is None:
        output = f"results/figures/cm_{run_name}.png"
    savefig(fig, output)
    plt.close(fig)

    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(true_labels, pred_labels,
                                 target_names=class_names, digits=4))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confusion matrix for a completed run")
    p.add_argument("--run-name",     required=True,
                   help="Run folder name under --results-dir")
    p.add_argument("--results-dir",  default="results/raw")
    p.add_argument("--no-normalize", action="store_true",
                   help="Show raw counts instead of row-normalised recall")
    p.add_argument("--output",       default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_confusion_matrix(
        run_name    = args.run_name,
        results_dir = args.results_dir,
        normalize   = not args.no_normalize,
        output      = args.output,
    )


if __name__ == "__main__":
    main()
