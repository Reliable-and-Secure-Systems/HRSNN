"""Snapshot of the software and hardware a run was produced with."""
from __future__ import annotations

import datetime
import importlib
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PACKAGES = ("torch", "torchvision", "snntorch", "numpy", "tonic")

_CODE_DIRS = ("src", "experiments", "config", "fault_injection", "hardware")


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _gpu_driver_version() -> Optional[str]:
    """Display-driver version, which torch does not expose."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip().splitlines()[0].strip()


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in _PACKAGES:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def collect_environment() -> Dict[str, Any]:
    """Record the Python, package, CUDA, GPU and git state of the current run."""
    status = _git("status", "--porcelain", "--", *_CODE_DIRS)
    dirty: Optional[List[str]] = None if status is None else [
        line for line in status.splitlines() if line.strip()
    ]
    return {
        "recorded_at":     datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python":          platform.python_version(),
        "platform":        platform.platform(),
        "packages":        _package_versions(),
        "cuda":            torch.version.cuda,
        "cudnn":           torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_driver":      _gpu_driver_version(),
        "git_commit":      _git("rev-parse", "HEAD"),
        "git_dirty_files": dirty,
    }
