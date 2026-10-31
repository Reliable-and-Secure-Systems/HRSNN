from __future__ import annotations

from fault_injection.transient.bit_flip         import BitFlipInjector, INT8BitFlipInjector
from fault_injection.transient.membrane_seu     import MembraneSEU
from fault_injection.transient.probabilistic_fi import ProbabilisticFaultInjector

TRANSIENT_REGISTRY = {
    "bit_flip":      BitFlipInjector,
    "int_bit_flip":  INT8BitFlipInjector,
    "membrane_seu":  MembraneSEU,
    "probabilistic": ProbabilisticFaultInjector,
}


def get_transient_injector(name: str, **kwargs):
    if name not in TRANSIENT_REGISTRY:
        raise ValueError(
            f"Unknown transient fault '{name}'. "
            f"Available: {list(TRANSIENT_REGISTRY)}"
        )
    return TRANSIENT_REGISTRY[name](**kwargs)
