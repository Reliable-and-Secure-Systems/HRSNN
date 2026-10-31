"""Run one SNN experiment from four YAML config files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import run_experiment, resolve_config_path
from src.utils.logging   import get_logger

logger = get_logger("run_single")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single SNN experiment")
    p.add_argument("--dataset",   default="config/datasets/mnist.yaml",
                   help="Dataset YAML (path or short name, e.g. 'mnist')")
    p.add_argument("--arch",      default="config/architectures/mlp_5layer.yaml",
                   help="Architecture YAML (path or short name, e.g. 'mlp_5layer')")
    p.add_argument("--coding",    default="config/coding_schemes/rate.yaml",
                   help="Coding-scheme YAML (path or short name, e.g. 'rate')")
    p.add_argument("--learning",  default="config/learning_rules/surrogate_gradient.yaml",
                   help="Learning-rule YAML (path or short name, e.g. 'surrogate_gradient')")
    p.add_argument("--run-name",  default=None,
                   help="Unique run identifier (auto-generated if omitted)")
    p.add_argument("--device",    default="auto",
                   help="'auto' | 'cpu' | 'cuda' | 'cuda:0'")
    p.add_argument("--synthetic", action="store_true",
                   help="Use random synthetic data — no download needed")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    result = run_experiment(
        ds_path     = resolve_config_path(args.dataset,  "dataset"),
        arch_path   = resolve_config_path(args.arch,     "arch"),
        coding_path = resolve_config_path(args.coding,   "coding"),
        lr_path     = resolve_config_path(args.learning, "learning"),
        run_name    = args.run_name,
        seed        = args.seed,
        device_str  = args.device,
        synthetic   = args.synthetic,
    )

    print(f"\nDone.  test_acc={result['test_acc']:.4f}  "
          f"best_val_acc={result['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
