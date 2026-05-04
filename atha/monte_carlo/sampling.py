# atha/monte_carlo/sampling.py
from __future__ import annotations
from typing import List
import numpy as np
from scipy.stats.qmc import LatinHypercube
from atha.monte_carlo.parameters import UncertainParameter


class LHSSampler:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample(self, params: List[UncertainParameter], N: int) -> np.ndarray:
        """Return (N, k) array in physical parameter space."""
        k = len(params)
        sampler = LatinHypercube(d=k, seed=self.seed)
        unit_samples = sampler.random(N)   # shape (N, k), values in [0, 1]

        physical = np.zeros_like(unit_samples)
        for j, param in enumerate(params):
            u = np.clip(unit_samples[:, j], 1e-10, 1 - 1e-10)
            physical[:, j] = np.array([param.transform_unit(ui) for ui in u])
        return physical


class SaltelliSampler:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample(self, params: List[UncertainParameter], N_base: int) -> np.ndarray:
        """Return (N_base*(k+2), k) array in physical parameter space."""
        from SALib.sample import saltelli

        k = len(params)
        problem = {
            "num_vars": k,
            "names": [p.name for p in params],
            "bounds": [[0.0, 1.0] for _ in range(k)],
        }
        # Set numpy seed for reproducibility (saltelli uses numpy's RNG internally)
        rng_state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            unit_samples = saltelli.sample(
                problem, N=N_base, calc_second_order=False
            )
        finally:
            np.random.set_state(rng_state)

        physical = np.zeros_like(unit_samples)
        for j, param in enumerate(params):
            u = np.clip(unit_samples[:, j], 1e-10, 1 - 1e-10)
            physical[:, j] = np.array([param.transform_unit(ui) for ui in u])
        return physical
