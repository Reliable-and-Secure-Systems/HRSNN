from __future__ import annotations

from fault_injection.permanent.memory_block     import MemoryBlockStuckAt
from fault_injection.permanent.memristor_stuck  import MemristorStuckAt
from fault_injection.permanent.stuck_at_weights import IntStuckAtWeights, StuckAtWeights
from fault_injection.permanent.stuck_at_logic   import StuckAtLogic

PERMANENT_REGISTRY = {
    "memory_block":         MemoryBlockStuckAt,
    "memristor_stuck":      MemristorStuckAt,
    "int_stuck_at_weights": IntStuckAtWeights,
    "stuck_at_weights":     StuckAtWeights,
    "stuck_at_logic":       StuckAtLogic,
}


def get_permanent_injector(name: str, **kwargs):
    if name not in PERMANENT_REGISTRY:
        raise ValueError(
            f"Unknown permanent fault '{name}'. "
            f"Available: {list(PERMANENT_REGISTRY)}"
        )
    return PERMANENT_REGISTRY[name](**kwargs)
