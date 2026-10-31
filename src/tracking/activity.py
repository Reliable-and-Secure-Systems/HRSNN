"""Spike activity and synaptic-operation (SynOps) counts for a trained model."""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def measure_activity(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """One no-grad pass over `loader` counting spikes into every nn.Linear layer."""
    linear = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    counts = {name: torch.zeros((), dtype=torch.int64, device=device)
              for name, _ in linear}
    elems  = {name: 0 for name, _ in linear}

    def _make_hook(name: str):
        def hook(module, inp, out):
            x = inp[0].detach()
            counts[name] += (x.abs() > 0.5).sum()
            elems[name]  += x.numel()
        return hook

    handles = [m.register_forward_hook(_make_hook(n)) for n, m in linear]

    n_samples  = 0
    time_steps = 0
    correct_t: Optional[torch.Tensor] = None
    decision:  Optional[torch.Tensor] = None

    batches = itertools.islice(loader, max_batches) if max_batches is not None else loader
    model.eval()
    try:
        with torch.no_grad():
            for spikes, labels in batches:
                spikes = spikes.to(device).permute(1, 0, 2).contiguous()
                labels = labels.to(device).long()

                spk_rec, _ = model(spikes)
                preds      = spk_rec.cumsum(dim=0).argmax(dim=2)
                hits       = (preds == labels.unsqueeze(0)).sum(dim=1)
                correct_t  = hits if correct_t is None else correct_t + hits

                T          = preds.shape[0]
                trailing   = (preds == preds[-1:]).flip(0).int().cumprod(0).sum(0)
                step       = T - trailing
                right      = preds[-1] == labels
                hist       = torch.bincount(step[right], minlength=T)
                decision   = hist if decision is None else decision + hist

                n_samples  += labels.size(0)
                time_steps  = T
    finally:
        for h in handles:
            h.remove()

    n        = max(n_samples, 1)
    n_spikes = {name: int(c.item()) for name, c in counts.items()}
    layers: List[Dict[str, Any]] = []
    for name, module in linear:
        fan_out, fan_in = module.weight.shape
        spikes_l = n_spikes[name]
        layers.append({
            "name":               name,
            "fan_in":             int(fan_in),
            "fan_out":            int(fan_out),
            "spikes_per_sample":  spikes_l / n,
            "spike_rate":         spikes_l / max(elems[name], 1),
            "syn_ops_per_sample": spikes_l * fan_out / n,
        })

    total_spikes = sum(n_spikes.values())
    total_elems  = sum(elems.values())
    syn_ops      = sum(l["syn_ops_per_sample"] for l in layers)
    ann_macs     = float(sum(l["fan_in"] * l["fan_out"] for l in layers))

    return {
        "n_samples":          n_samples,
        "time_steps":         time_steps,
        "mean_spike_rate":    total_spikes / max(total_elems, 1),
        "spikes_per_sample":  total_spikes / n,
        "syn_ops_per_sample": syn_ops,
        "ann_macs":           ann_macs,
        "efficiency_ratio":   syn_ops / max(ann_macs, 1.0),
        "acc_by_timestep":    ([] if correct_t is None
                               else (correct_t.double() / n).tolist()),
        "decision_step_hist": [] if decision is None else decision.tolist(),
        "layers":             layers,
    }
