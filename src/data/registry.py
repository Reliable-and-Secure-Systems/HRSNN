"""Dataset registry — image datasets require an encoder; DVS datasets do not."""
from __future__ import annotations
import importlib

_DATASET_REGISTRY: dict[str, tuple[str, str, bool]] = {
    "mnist":       ("src.data.mnist",       "MNISTDataset",   True),
    "cifar10":     ("src.data.cifar10",     "CIFAR10Dataset", True),
    "nmnist":      ("src.data.nmnist",      "NMNISTDataset",  False),
}

DATASET_REGISTRY = {k: v[1] for k, v in _DATASET_REGISTRY.items()}


def needs_encoder(name: str) -> bool:
    """Return True if this dataset requires a spike encoder (image dataset)."""
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_DATASET_REGISTRY)}")
    return _DATASET_REGISTRY[name][2]


def get_dataset(name: str, **kwargs):
    """Instantiate a dataset by name."""
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_DATASET_REGISTRY)}")
    module_path, class_name, _ = _DATASET_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
