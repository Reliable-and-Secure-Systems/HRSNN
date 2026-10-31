from __future__ import annotations
from src.models.mlp       import SNNMLP


MODEL_REGISTRY: dict = {
    "mlp":       SNNMLP,
}


def get_model(name: str, cfg, input_dim: int, n_classes: int, time_steps: int):
    """Instantiate a model from its YAML config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    if name == "mlp":
        return SNNMLP(
            input_dim       = input_dim,
            n_classes       = n_classes,
            hidden_dims     = cfg.get("hidden_dims",     [512, 256, 128, 64]),
            neuron_type     = cfg.get("neuron_type",     "lif"),
            beta            = cfg.get("beta",            0.9),
            threshold       = cfg.get("threshold",       1.0),
            dropout         = cfg.get("dropout",         0.2),
            surrogate_slope = cfg.get("surrogate_slope", 25.0),
            time_steps      = time_steps,
        )

    raise NotImplementedError(f"Builder for '{name}' not yet wired.")
