"""Shared utilities for all analysis scripts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib  import Path
from typing   import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SCHEME_ORDER = ["rate", "ttfs", "phase", "burst", "population"]
RULE_ORDER   = ["surrogate_gradient", "bptt", "stdp", "eprop"]

SCHEME_COLORS: dict[str, str] = {
    "rate":       "#1f77b4",
    "ttfs":       "#ff7f0e",
    "phase":      "#d62728",
    "burst":      "#9467bd",
    "population": "#8c564b",
}

RULE_COLORS: dict[str, str] = {
    "surrogate_gradient": "#1f77b4",
    "bptt":               "#ff7f0e",
    "stdp":               "#2ca02c",
    "eprop":              "#d62728",
}

SCHEME_LABELS: dict[str, str] = {
    "rate":       "Rate Coding",
    "ttfs":       "TTFS Coding",
    "phase":      "Phase Coding",
    "burst":      "Burst Coding",
    "population": "Population Coding",
}

RULE_LABELS: dict[str, str] = {
    "surrogate_gradient": "Surrogate Gradient",
    "bptt":               "BPTT",
    "stdp":               "STDP",
    "eprop":              "E-Prop",
}


def load_all_runs(results_dir: str | Path = "results/raw") -> list[dict[str, Any]]:
    """Scan results_dir and return one summary dict per completed run."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        entry = _load_run_meta(run_dir)
        if entry:
            runs.append(entry)
    return runs


def load_epoch_history(run_dir: Path | str) -> list[dict]:
    """Load per-epoch metrics from metrics.csv in a run directory."""
    path = Path(run_dir) / "metrics.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    result = []
    for r in rows:
        row = {}
        for k, v in r.items():
            try:
                row[k] = int(v) if k == "epoch" else float(v)
            except (ValueError, TypeError):
                row[k] = v
        result.append(row)

    if not result:
        return result

    sequences: list[list[dict]] = []
    current = [result[0]]
    for i in range(1, len(result)):
        if result[i].get("epoch", i + 1) <= result[i - 1].get("epoch", i):
            sequences.append(current)
            current = [result[i]]
        else:
            current.append(result[i])
    sequences.append(current)

    max_len = max(len(s) for s in sequences)
    return next(s for s in reversed(sequences) if len(s) == max_len)


def group_runs(
    runs: list[dict[str, Any]],
    by:   str = "coding",
) -> dict[str, list[dict[str, Any]]]:
    """Group a list of run dicts by any top-level key (coding, learning, dataset, arch)."""
    groups: dict[str, list] = {}
    for r in runs:
        key = str(r.get(by, "unknown"))
        groups.setdefault(key, []).append(r)
    return groups


def _load_run_meta(run_dir: Path) -> dict[str, Any] | None:
    """Build a summary dict for one run directory."""
    entry: dict[str, Any] = {"run_name": run_dir.name, "run_dir": str(run_dir)}

    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        entry["config"] = cfg
        entry["seed"]    = cfg.get("seed", 0)
        entry["input_dim"]  = cfg.get("input_dim", 0)
        entry["n_classes"]  = cfg.get("n_classes", 10)

        for section in ("dataset", "coding", "learning", "arch"):
            sub = cfg.get(section, {})
            if isinstance(sub, dict):
                entry[section] = sub.get("name", "")

        if "learning_rule" not in entry:
            entry["learning_rule"] = entry.get("learning", "")

    test_path = run_dir / "test_metrics.json"
    if test_path.exists():
        with open(test_path) as f:
            entry.update(json.load(f))
        if "acc"       in entry and "test_acc"  not in entry:
            entry["test_acc"]  = entry.pop("acc")
        if "loss"      in entry and "test_loss" not in entry:
            entry["test_loss"] = entry.pop("loss")

    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        with open(metrics_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            try:
                entry["best_val_acc"] = max(float(r["val_acc"]) for r in rows)
                entry["n_epochs"]     = len(rows)
            except (KeyError, ValueError):
                pass
    return entry


def set_plot_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "legend.framealpha":  0.9,
        "figure.dpi":         100,
        "savefig.dpi":        150,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    2.0,
        "lines.markersize":   5,
    })


def savefig(
    fig,
    path:    str | Path,
    dpi:     int  = 150,
    formats: tuple = (".png", ".pdf"),
) -> None:
    """Save figure to PNG and PDF at the given base path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for ext in formats:
        fig.savefig(str(stem) + ext, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {stem}.png  |  {stem}.pdf")


def rebuild_pipeline(
    run_dir:          str | Path,
    device_str:       str  = "auto",
    batch_size:       int  = 128,
    encode_on_device: bool = False,
):
    """Reconstruct model + test DataLoader from a completed run directory."""
    import torch
    from torch.utils.data import DataLoader

    from src.utils.config    import ConfigDict
    from src.utils.device    import get_device
    from src.coding.registry import get_encoder
    from src.data.registry   import get_dataset, needs_encoder
    from src.models.registry import get_model

    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        raw = json.load(f)

    ds_cfg     = ConfigDict(raw.get("dataset",  {}))
    arch_cfg   = ConfigDict(raw.get("arch",     {}))
    coding_cfg = ConfigDict(raw.get("coding",   {}))

    device     = get_device(device_str)
    ds_name    = ds_cfg.get("name", "mnist")
    cod_name   = coding_cfg.get("name", "rate")
    time_steps = coding_cfg.get("time_steps", 20)
    input_dim  = raw.get("input_dim", 784)
    n_classes  = raw.get("n_classes", 10)

    if raw.get("synthetic", False):
        raise ValueError(
            "Cannot rebuild pipeline for a synthetic run — no real dataset."
        )

    if needs_encoder(ds_name):
        encoder = get_encoder(cod_name, **{
            k: v for k, v in coding_cfg.to_dict().items()
            if k not in ("name", "time_steps")
        })
        test_ds = get_dataset(ds_name, train=False,
                              encoder=encoder, time_steps=time_steps,
                              data_dir=ds_cfg.get("data_dir", "data/"))
    else:
        test_ds = get_dataset(ds_name, train=False,
                              data_dir=ds_cfg.get("data_dir", "data/"),
                              time_steps=time_steps)

    if encode_on_device and needs_encoder(ds_name):
        from src.data.loaders import DeviceEncodedLoader
        test_loader = DeviceEncodedLoader(test_ds._images, test_ds._labels, encoder,
                                          time_steps, batch_size, device)
    else:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(arch_cfg.get("name", "mlp"), arch_cfg,
                      input_dim=input_dim, n_classes=n_classes, time_steps=time_steps)

    ckpt_path = run_dir / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"Warning: no checkpoint at {ckpt_path}; using random weights.")

    model = model.to(device).eval()
    return model, test_loader, device, raw


def collect_predictions(
    model,
    test_loader,
    device,
    max_batches: int | None = None,
):
    """Run inference and collect true labels, predicted labels, and class probs."""
    import torch
    import torch.nn.functional as F

    true_labels, pred_labels, probs = [], [], []

    with torch.no_grad():
        for i, (spikes, labels) in enumerate(test_loader):
            if max_batches and i >= max_batches:
                break
            spikes = spikes.to(device).permute(1, 0, 2).contiguous()
            labels = labels.to(device).long()

            spk_rec, _ = model(spikes)
            logits      = spk_rec.sum(dim=0)
            p           = F.softmax(logits, dim=1)

            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(logits.argmax(dim=1).cpu().tolist())
            probs.extend(p.cpu().tolist())

    return true_labels, pred_labels, probs
