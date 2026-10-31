from __future__ import annotations

from hardware.digital_imc          import (
    DigitalIMCTransientFault, DigitalIMCMembraneFault, DigitalIMCBlockFault,
)
from hardware.analog_imc           import AnalogIMCStuckOnFault, AnalogIMCStuckOffFault
from hardware.digital_neuromorphic import (
    DigitalNeuromorphicTransientFault, DigitalNeuromorphicMembraneFault,
    DigitalNeuromorphicBlockFault, LoihiWeightQuantization,
)
from hardware.analog_neuromorphic  import (
    AnalogNeuromorphicStuckOnFault, AnalogNeuromorphicStuckOffFault,
)

HARDWARE_REGISTRY = {
    "digital_imc":            DigitalIMCTransientFault,
    "digital_neuro":          DigitalNeuromorphicTransientFault,
    "digital_imc_membrane":   DigitalIMCMembraneFault,
    "digital_neuro_membrane": DigitalNeuromorphicMembraneFault,
    "digital_imc_block":      DigitalIMCBlockFault,
    "digital_neuro_block":    DigitalNeuromorphicBlockFault,
    "analog_imc_stuck_on":    AnalogIMCStuckOnFault,
    "analog_imc_stuck_off":   AnalogIMCStuckOffFault,
    "analog_neuro_stuck_on":  AnalogNeuromorphicStuckOnFault,
    "analog_neuro_stuck_off": AnalogNeuromorphicStuckOffFault,
    "loihi_quant":   LoihiWeightQuantization,
}


def get_hardware_model(name: str, **kwargs):
    if name not in HARDWARE_REGISTRY:
        raise ValueError(
            f"Unknown hardware model '{name}'. "
            f"Available: {list(HARDWARE_REGISTRY)}"
        )
    return HARDWARE_REGISTRY[name](**kwargs)
