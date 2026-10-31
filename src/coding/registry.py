from __future__ import annotations
from src.coding.rate       import RateEncoder
from src.coding.ttfs       import TTFSEncoder
from src.coding.phase      import PhaseEncoder
from src.coding.burst      import BurstEncoder
from src.coding.population import PopulationEncoder

ENCODER_REGISTRY: dict = {
    "rate":       RateEncoder,
    "ttfs":       TTFSEncoder,
    "phase":      PhaseEncoder,
    "burst":      BurstEncoder,
    "population": PopulationEncoder,
}


def get_encoder(name: str, **kwargs) -> object:
    """Instantiate an encoder by name."""
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](**kwargs)
