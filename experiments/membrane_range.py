"""Measure the operating membrane range of the trained grid, to size the register."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import rebuild_pipeline
from fault_injection.transient.bit_flip import IntWeightFaultInjector
from hardware import digital_imc, digital_neuromorphic
from src.utils.environment import collect_environment

SUBSTRATES = {
    "float":         None,
    "digital_imc":   digital_imc,
    "digital_neuro": digital_neuromorphic,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Operating membrane range of the trained grid")
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--output", default="results/aggregated/membrane_range.json")
    return p.parse_args()


def measure(model, loader, device, theta: float) -> dict:
    """Largest |membrane| / θ and share of neuron-steps above 2θ, fault-free."""
    stats = {"n": 0, "over_2": 0, "max": 0.0}

    def hook(block, inp, out):
        mem = out[1].detach().abs() / theta
        stats["n"] += mem.numel()
        stats["over_2"] += (mem > 2).sum().item()
        stats["max"] = max(stats["max"], mem.max().item())

    handles = [b.register_forward_hook(hook) for b in model.blocks]
    try:
        with torch.no_grad():
            for x, _ in loader:
                model(x.to(device).permute(1, 0, 2).contiguous())
    finally:
        for h in handles:
            h.remove()
    return {"max_abs_theta": round(stats["max"], 3),
            "share_above_2theta": round(stats["over_2"] / stats["n"], 4)}


def main() -> None:
    args = _parse_args()
    runs = sorted(p for p in Path(args.results_dir).iterdir()
                  if (p / "test_metrics.json").exists() and (p / "best_model.pth").exists())
    out = {"rule": "smallest power of two >= largest |membrane|/theta on a digital substrate",
           "environment": collect_environment(), "runs": {}}

    for i, run_dir in enumerate(runs, 1):
        torch.manual_seed(0)
        model, loader, device, cfg = rebuild_pipeline(run_dir, encode_on_device=True)
        theta = cfg["arch"].get("threshold", 1.0)
        per = {}
        for name, module in SUBSTRATES.items():
            torch.manual_seed(0)
            if module is None:
                per[name] = measure(model, loader, device, theta)
                continue
            sub = IntWeightFaultInjector(n_sites=0, bits=module.BITS,
                                         weight_noise_std=module.WEIGHT_NOISE_STD)
            sub.inject(model)
            try:
                per[name] = measure(model, loader, device, theta)
            finally:
                sub.restore(model)
        out["runs"][run_dir.name] = per
        print(f"[{i}/{len(runs)}] {run_dir.name}: "
              + "  ".join(f"{k} {v['max_abs_theta']:.1f}" for k, v in per.items()), flush=True)

    digital = [v[s]["max_abs_theta"] for v in out["runs"].values()
               for s in ("digital_imc", "digital_neuro")]
    largest = max(digital)
    out["max_abs_theta_digital"] = largest
    out["register_range"] = 2 ** math.ceil(math.log2(largest))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"largest |membrane| on a digital substrate: {largest:.1f} theta "
          f"-> register range +-{out['register_range']} theta  ({out_path})")


if __name__ == "__main__":
    main()
