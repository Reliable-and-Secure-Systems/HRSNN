from __future__ import annotations
from src.learning.surrogate_gradient import SurrogateGradientLearner
from src.learning.stdp               import STDPLearner, RewardModulatedSTDP
from src.learning.bptt               import BPTTLearner
from src.learning.eprop              import EPropLearner

LEARNER_REGISTRY: dict = {
    "surrogate":           SurrogateGradientLearner,
    "surrogate_gradient":  SurrogateGradientLearner,
    "stdp":                STDPLearner,
    "r_stdp":              RewardModulatedSTDP,
    "bptt":                BPTTLearner,
    "eprop":               EPropLearner,
    "e_prop":              EPropLearner,
}


def get_learner(name: str, **kwargs):
    """Instantiate a learner by registry name."""
    if name not in LEARNER_REGISTRY:
        raise ValueError(f"Unknown learner '{name}'. Available: {list(LEARNER_REGISTRY)}")
    return LEARNER_REGISTRY[name](**kwargs)
