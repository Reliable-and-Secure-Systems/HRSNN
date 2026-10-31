"""Evaluate trained SNN checkpoints under hardware fault models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._common import rebuild_pipeline, load_all_runs
from fault_injection.framework import evaluate_under_fault
from hardware.digital_imc          import (
    DigitalIMCTransientFault, DigitalIMCMembraneFault, DigitalIMCBlockFault,
)
from hardware.digital_neuromorphic import (
    DigitalNeuromorphicTransientFault, DigitalNeuromorphicMembraneFault,
    DigitalNeuromorphicBlockFault,
)
from hardware.analog_imc           import AnalogIMCStuckOnFault, AnalogIMCStuckOffFault
from hardware.analog_neuromorphic  import (
    AnalogNeuromorphicStuckOnFault, AnalogNeuromorphicStuckOffFault,
)
from src.training.trainer import evaluate as eval_clean
from src.utils.environment import collect_environment
from src.utils.logging import get_logger

logger = get_logger("run_fault_evaluation")


WEIGHT_BER       = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
MEMBRANE_RATE    = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
BLOCK_FRACTION   = [1e-4, 1e-3, 1e-2]
STUCK_CELL_FRAC  = [1e-3, 1e-2, 5e-2, 1e-1]

FAULT_MODEL_VERSION = "2026-10-10"
CATEGORIES          = ("transient", "permanent")


def _sweep(category, hardware, site, label, param, values, build):
    return {"category": category, "hardware": hardware, "site": site, "label": label,
            "param": param, "values": values, "logscale": True, "build": build}


FAULT_SWEEPS: dict[str, dict] = {
    "digital_imc": _sweep(
        "transient", "digital_imc", "weight_memory", "Digital IMC (4-bit SRAM)",
        "ber", WEIGHT_BER,
        lambda n, s: DigitalIMCTransientFault(n_flips=n, seed=s)),
    "digital_imc_membrane": _sweep(
        "transient", "digital_imc", "neuron_state", "Digital IMC, membrane registers",
        "upsets_per_bit_step", MEMBRANE_RATE,
        lambda n, s: DigitalIMCMembraneFault(n_upsets=n, seed=s)),
    "digital_imc_block": _sweep(
        "permanent", "digital_imc", "weight_memory_blocks", "Digital IMC (4-bit SRAM), blocks",
        "faulty_bit_fraction", BLOCK_FRACTION,
        lambda n, s: DigitalIMCBlockFault(n_faulty_bits=n, seed=s)),
    "digital_neuromorphic": _sweep(
        "transient", "digital_neuro", "weight_memory", "Digital Neuromorphic (INT8)",
        "ber", WEIGHT_BER,
        lambda n, s: DigitalNeuromorphicTransientFault(n_flips=n, seed=s)),
    "digital_neuromorphic_membrane": _sweep(
        "transient", "digital_neuro", "neuron_state", "Digital Neuromorphic, membrane registers",
        "upsets_per_bit_step", MEMBRANE_RATE,
        lambda n, s: DigitalNeuromorphicMembraneFault(n_upsets=n, seed=s)),
    "digital_neuromorphic_block": _sweep(
        "permanent", "digital_neuro", "weight_memory_blocks", "Digital Neuromorphic (INT8), blocks",
        "faulty_bit_fraction", BLOCK_FRACTION,
        lambda n, s: DigitalNeuromorphicBlockFault(n_faulty_bits=n, seed=s)),
    "analog_imc_stuck_on": _sweep(
        "permanent", "analog_imc", "memristor_cells", "Analog IMC, cells stuck ON",
        "stuck_cell_fraction", STUCK_CELL_FRAC,
        lambda n, s: AnalogIMCStuckOnFault(n_cells=n, seed=s)),
    "analog_imc_stuck_off": _sweep(
        "permanent", "analog_imc", "memristor_cells", "Analog IMC, cells stuck OFF",
        "stuck_cell_fraction", STUCK_CELL_FRAC,
        lambda n, s: AnalogIMCStuckOffFault(n_cells=n, seed=s)),
    "analog_neuromorphic_stuck_on": _sweep(
        "permanent", "analog_neuro", "memristor_cells", "Analog Neuromorphic, cells stuck ON",
        "stuck_cell_fraction", STUCK_CELL_FRAC,
        lambda n, s: AnalogNeuromorphicStuckOnFault(n_cells=n, seed=s)),
    "analog_neuromorphic_stuck_off": _sweep(
        "permanent", "analog_neuro", "memristor_cells", "Analog Neuromorphic, cells stuck OFF",
        "stuck_cell_fraction", STUCK_CELL_FRAC,
        lambda n, s: AnalogNeuromorphicStuckOffFault(n_cells=n, seed=s)),
}

ALL_FAULTS = list(FAULT_SWEEPS.keys())


def _load_sweeps(path: Path) -> dict[str, Any]:
    """Sweeps already stored in a fault_results.json ({} if none)."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("fault_sweeps", {})


def _pending_sweeps(path: Path, fault_names: list[str], overwrite: bool,
                    n_trials: int, max_batches: int | None) -> list[str]:
    """Return the fault sweeps of one run that are missing or were stored with other settings."""
    if overwrite:
        return list(fault_names)
    stored = _load_sweeps(path)
    pending = []
    for name in fault_names:
        sw = stored.get(name, {})
        if (sw.get("fault_model_version") != FAULT_MODEL_VERSION
                or sw.get("n_trials") != n_trials
                or sw.get("max_batches") != max_batches):
            pending.append(name)
    return pending


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate trained SNN models under hardware fault injection"
    )
    p.add_argument("--results-dir", default="results/raw",
                   help="Directory containing run subdirectories")
    p.add_argument("--dataset",     default=None,
                   help="Filter by dataset name (mnist | cifar10)")
    p.add_argument("--codings",     nargs="+", default=None, metavar="CODING",
                   help="Filter by coding scheme names")
    p.add_argument("--category",    choices=CATEGORIES, default=None,
                   help="Run only transient or only permanent sweeps (default: both)")
    p.add_argument("--faults",      nargs="+", default=None, metavar="FAULT",
                   help=f"Sweep names to run. Available: {ALL_FAULTS}")
    p.add_argument("--n-trials",    type=int, default=10,
                   help="Independent fault trials per intensity point (default: 10)")
    p.add_argument("--device",      default="auto")
    p.add_argument("--cpu-encoding", action="store_true",
                   help="Encode spikes per image on the CPU instead of per batch on the device")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap test batches per trial for speed (default: full test set). "
                        "E.g. --max-batches 10 evaluates ~1000 MNIST samples per trial "
                        "instead of 10000 — ~10x faster, accuracy estimate within ±1%%.")
    p.add_argument("--overwrite",   action="store_true",
                   help="Recompute every selected sweep, even if already present")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print planned work without running anything")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    fault_names = args.faults or ALL_FAULTS
    unknown = [f for f in fault_names if f not in FAULT_SWEEPS]
    if unknown:
        raise ValueError(f"Unknown fault type(s): {unknown}. Available: {ALL_FAULTS}")
    if args.category:
        fault_names = [f for f in fault_names
                       if FAULT_SWEEPS[f]["category"] == args.category]

    runs = load_all_runs(args.results_dir)
    if args.dataset:
        runs = [r for r in runs if r.get("dataset") == args.dataset]
    if args.codings:
        runs = [r for r in runs if r.get("coding") in args.codings]

    valid_runs = [
        r for r in runs
        if (Path(r["run_dir"]) / "best_model.pth").exists()
        and (Path(r["run_dir"]) / "test_metrics.json").exists()
        and not r.get("config", {}).get("synthetic", False)
    ]

    total_points = sum(len(FAULT_SWEEPS[f]["values"]) for f in fault_names)
    print(f"Finished runs         : {len(valid_runs)}")
    print(f"Fault types           : {fault_names}")
    print(f"Intensity points total: {total_points}")
    print(f"Trials per point      : {args.n_trials}")
    print(f"Total evaluations     : {len(valid_runs) * total_points * args.n_trials}\n")

    if args.dry_run:
        for r in valid_runs:
            todo = _pending_sweeps(Path(r["run_dir"]) / "fault_results.json",
                                   fault_names, args.overwrite,
                                   args.n_trials, args.max_batches)
            status = f"{len(todo)} pending" if todo else "done"
            print(f"  {r['run_name']:<55}  [{status}]")
        return

    for i, run in enumerate(valid_runs, 1):
        run_dir  = Path(run["run_dir"])
        run_name = run["run_name"]
        out_path = run_dir / "fault_results.json"

        todo = _pending_sweeps(out_path, fault_names, args.overwrite,
                               args.n_trials, args.max_batches)
        if not todo:
            print(f"[{i}/{len(valid_runs)}]  {run_name}  — skipped (already done)")
            continue

        print(f"\n[{i}/{len(valid_runs)}]  {run_name}")
        print(f"  dataset={run.get('dataset','')}  coding={run.get('coding','')}"
              f"  learning={run.get('learning','')}  seed={run.get('seed','')}")

        try:
            model, test_loader, device, _ = rebuild_pipeline(
                run_dir, device_str=args.device,
                encode_on_device=not args.cpu_encoding
            )
        except Exception as exc:
            logger.error("  rebuild failed: %s", exc)
            continue

        time_steps = run.get("config", {}).get("coding", {}).get("time_steps", 20)

        torch.manual_seed(42)
        baseline_metrics = eval_clean(model, test_loader, device)
        baseline_acc     = baseline_metrics["acc"]
        print(f"  baseline_acc = {baseline_acc:.4f}")

        fault_sweeps_out: dict[str, Any] = {
            name: sweep for name, sweep in _load_sweeps(out_path).items()
            if name not in todo
        }
        substrate_accs: dict[str, float] = {}
        environment = collect_environment()

        for fault_name in todo:
            sweep    = FAULT_SWEEPS[fault_name]
            pts: list[dict] = []

            substrate_inj = sweep["build"](0, None)
            hw = sweep["hardware"]
            if hw not in substrate_accs:
                substrate_accs[hw] = evaluate_under_fault(
                    model        = model,
                    loader       = test_loader,
                    device       = device,
                    injector     = substrate_inj,
                    n_trials     = args.n_trials,
                    baseline_acc = baseline_acc,
                    seed         = 42,
                    max_batches  = args.max_batches,
                )["mean_acc"]
            substrate_acc = substrate_accs[hw]
            print(f"    {fault_name:<30} substrate_acc = {substrate_acc:.4f}"
                  f"  (vs float32 baseline {baseline_acc:.4f})")

            total_sites = substrate_inj.fault_sites(model, time_steps)

            for rate in sweep["values"]:
                n_faults = max(1, round(rate * total_sites))
                injector = sweep["build"](n_faults, None)
                result   = evaluate_under_fault(
                    model        = model,
                    loader       = test_loader,
                    device       = device,
                    injector     = injector,
                    n_trials     = args.n_trials,
                    baseline_acc = substrate_acc,
                    seed         = 42,
                    max_batches  = args.max_batches,
                )
                point = {
                    "intensity":   rate,
                    "n_faults":    n_faults,
                    "total_sites": total_sites,
                    "mean_acc":    result["mean_acc"],
                    "std_acc":     result["std_acc"],
                    "min_acc":     result["min_acc"],
                    "max_acc":     result["max_acc"],
                    "degradation": result["degradation"],
                }
                if hasattr(injector, "fault_stats"):
                    st = injector.fault_stats()
                    point["placed"] = st.get("placed")
                    point["realised_fraction"] = (st.get("placed") or 0) / total_sites
                    if "changed" in st:
                        point["changed"] = st["changed"]
                        point["effective_fraction"] = st["changed"] / total_sites
                    if "clamped_fraction" in st:
                        point["clamped_fraction"] = st["clamped_fraction"]
                pts.append(point)
                print(
                    f"    {fault_name:<30} rate={rate:.0e}  n_faults={n_faults:<8}"
                    f"  acc={result['mean_acc']:.4f}±{result['std_acc']:.4f}"
                    f"  deg={result['degradation']:.4f}"
                )

            fault_sweeps_out[fault_name] = {
                "category":        sweep["category"],
                "hardware":        sweep["hardware"],
                "site":            sweep["site"],
                "label":           sweep["label"],
                "param_name":      sweep["param"],
                "logscale":        sweep["logscale"],
                "fault_type":      substrate_inj.fault_type,
                "fault_model_version": FAULT_MODEL_VERSION,
                "n_trials":        args.n_trials,
                "max_batches":     args.max_batches,
                "substrate_acc":   substrate_acc,
                "environment":     environment,
                "results":         pts,
            }

        output = {
            "run_name":     run_name,
            "dataset":      run.get("dataset",  ""),
            "coding":       run.get("coding",   ""),
            "learning":     run.get("learning", ""),
            "arch":         run.get("arch",     ""),
            "seed":         run.get("seed",     0),
            "baseline_acc": baseline_acc,
            "environment":  collect_environment(),
            "fault_sweeps": fault_sweeps_out,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  saved -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
