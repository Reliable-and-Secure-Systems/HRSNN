"""How much do membrane-upset results depend on the register range?"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import rebuild_pipeline
from experiments.run_fault_evaluation import MEMBRANE_RATE
from fault_injection.framework import evaluate_under_fault
from fault_injection.transient.bit_flip import IntWeightFaultInjector
from fault_injection.transient.membrane_seu import MembraneSEU, REGISTER_BITS
from hardware import digital_imc, digital_neuromorphic
from src.utils.environment import collect_environment

SUBSTRATES = {"digital_neuro": digital_neuromorphic, "digital_imc": digital_imc}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Membrane-register range sensitivity")
    p.add_argument("--run", required=True)
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--ranges", nargs="+", type=int, default=[64, 128, 256, 512])
    p.add_argument("--substrate", choices=list(SUBSTRATES), default="digital_neuro")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.results_dir) / args.run
    out_path = Path(args.output or f"results/aggregated/membrane_sensitivity_{args.run}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "run_name": args.run, "substrate": args.substrate, "n_trials": args.n_trials,
        "register_bits": REGISTER_BITS, "rates": MEMBRANE_RATE, "ranges": {},
    }

    module = SUBSTRATES[args.substrate]
    model, loader, device, cfg = rebuild_pipeline(run_dir, encode_on_device=True)
    time_steps = cfg["coding"].get("time_steps", 20)

    def substrate():
        return IntWeightFaultInjector(n_sites=0, bits=module.BITS,
                                      weight_noise_std=module.WEIGHT_NOISE_STD)

    base = evaluate_under_fault(model, loader, device, substrate(),
                                n_trials=args.n_trials, seed=42)["mean_acc"]
    out["substrate_acc"] = base
    print(f"{args.run}  substrate={args.substrate}  substrate_acc={base:.4f}")

    for rng in args.ranges:
        if str(rng) in out["ranges"]:
            print(f"  range ±{rng}θ: already done, skipped")
            continue
        sites = MembraneSEU(register_range=rng).fault_sites(model, time_steps)
        points = []
        for rate in MEMBRANE_RATE:
            n = max(1, round(rate * sites))
            inj = MembraneSEU(n_upsets=n, register_range=rng, substrate=substrate())
            res = evaluate_under_fault(model, loader, device, inj, n_trials=args.n_trials,
                                       baseline_acc=base, seed=42)
            st = inj.fault_stats()
            points.append({"intensity": rate, "n_faults": n, "mean_acc": res["mean_acc"],
                           "std_acc": res["std_acc"], "degradation": res["degradation"],
                           "clamped_fraction": st["clamped_fraction"]})
            print(f"  range ±{rng:<4}θ rate={rate:.0e} n={n:<6} acc={res['mean_acc']:.4f}"
                  f" deg={res['degradation']:+.4f} clamped={st['clamped_fraction']:.4%}")
        out["ranges"][str(rng)] = {"results": points, "environment": collect_environment()}
        out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
