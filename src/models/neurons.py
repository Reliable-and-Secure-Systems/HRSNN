from __future__ import annotations
import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate
    _HAS_SNNTORCH = True
except ImportError:
    _HAS_SNNTORCH = False


def _check_snntorch():
    if not _HAS_SNNTORCH:
        raise ImportError("snntorch required:  pip install snntorch")


def build_lif(
    beta: float           = 0.9,
    threshold: float      = 1.0,
    reset_mechanism: str  = "subtract",
    surrogate_slope: float = 25.0,
):
    """Standard LIF neuron."""
    _check_snntorch()
    sg = surrogate.fast_sigmoid(slope=surrogate_slope)
    return snn.Leaky(
        beta=beta,
        threshold=threshold,
        reset_mechanism=reset_mechanism,
        spike_grad=sg,
    )


def build_alif(
    beta: float           = 0.9,
    threshold: float      = 1.0,
    surrogate_slope: float = 25.0,
):
    """Adaptive LIF approximation using snn.Leaky with subtractive reset."""
    _check_snntorch()
    sg = surrogate.fast_sigmoid(slope=surrogate_slope)
    return snn.Leaky(
        beta=beta,
        threshold=threshold,
        reset_mechanism="subtract",
        spike_grad=sg,
    )


def build_plif(
    beta_init: float      = 0.9,
    threshold: float      = 1.0,
    surrogate_slope: float = 25.0,
):
    """Parametric LIF: beta is a learnable parameter per neuron."""
    _check_snntorch()
    sg = surrogate.fast_sigmoid(slope=surrogate_slope)
    return snn.Leaky(
        beta=beta_init,
        threshold=threshold,
        learn_beta=True,
        spike_grad=sg,
    )


def build_glif(
    beta: float           = 0.9,
    threshold: float      = 1.0,
    surrogate_slope: float = 25.0,
):
    """Generalized LIF: multiple learnable time constants."""
    _check_snntorch()
    sg = surrogate.fast_sigmoid(slope=surrogate_slope)
    return snn.Leaky(
        beta=beta,
        threshold=threshold,
        learn_beta=True,
        learn_threshold=True,
        spike_grad=sg,
    )


NEURON_BUILDERS = {
    "lif":  build_lif,
    "alif": build_alif,
    "plif": build_plif,
    "glif": build_glif,
}


def get_neuron(name: str, **kwargs):
    if name not in NEURON_BUILDERS:
        raise ValueError(f"Unknown neuron '{name}'. Available: {list(NEURON_BUILDERS)}")
    return NEURON_BUILDERS[name](**kwargs)
