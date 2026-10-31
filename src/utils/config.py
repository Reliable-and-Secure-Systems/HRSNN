from __future__ import annotations
import copy
import yaml
from pathlib import Path
from typing import Any

class ConfigDict:
    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, ConfigDict(v) if isinstance(v, dict) else v)


    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, ConfigDict) else v
        return out

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self):
        return f"ConfigDict({self.to_dict()})"

    def __contains__(self, key):
        return hasattr(self, key)


def load_yaml(path: str | Path) -> ConfigDict:
    """"Load a single YAML file into a ConfigDict"""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
        return ConfigDict(data)

def merge_config(*configs: ConfigDict) -> ConfigDict:
    """Merge multiple ConfigDicts left to right"""

    merged = {}
    for cfg in configs:
        _deep_update(merged, cfg.to_dict())
    return ConfigDict(merged)

def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

def save_yaml(cfg: ConfigDict, path: str | Path):
    """Serialize a ConfigDict back to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)
