from .base import BaseGenerator
from .baselines import RandomGenerator, MarginalGenerator, GMMGenerator, SMOTEGenerator
from .synthcity_generators import CTGANGenerator, TVAEGenerator, PATEGANGenerator, DPGANGenerator
from .tabpfngen import TabPFNGenGenerator
from .tabsyn import TabSynGenerator

__all__ = [
    "BaseGenerator",
    "RandomGenerator",
    "MarginalGenerator",
    "GMMGenerator",
    "SMOTEGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
    "PATEGANGenerator",
    "DPGANGenerator",
    "TabPFNGenGenerator",
    "TabSynGenerator",
]

GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {
    "random":          RandomGenerator,
    "marginal":        MarginalGenerator,
    "gmm":             GMMGenerator,
    "smote":           SMOTEGenerator,
    "ctgan":           CTGANGenerator,
    "tvae":            TVAEGenerator,
    "pategan":         PATEGANGenerator,
    "dpgan":           DPGANGenerator,
    "tabpfngen":       TabPFNGenGenerator,
    "tabsyn":          TabSynGenerator,
}


def get_generator(name: str, **kwargs) -> BaseGenerator:
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator '{name}'. Available: {list(GENERATOR_REGISTRY)}")
    return GENERATOR_REGISTRY[name](**kwargs)
