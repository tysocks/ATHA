# atha/monte_carlo/runner.py
from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np
from joblib import Parallel, delayed

from atha.monte_carlo.parameters import UncertainParameter
from atha.monte_carlo.sampling import LHSSampler, SaltelliSampler
from atha.monte_carlo.statistics import compute_statistics


class MonteCarloRunner:
    def __init__(
        self,
        params: List[UncertainParameter],
        n_samples: int = 500,
        sampling_method: str = "lhs",
        sampler: Optional[str] = None,
        n_jobs: int = -1,
        seed: int = 42,
        verbose: int = 0,
        evaluate_fn=None,
    ):
        self.params = params
        self.n_samples = n_samples
        self.sampling_method = sampler if sampler is not None else sampling_method
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self._evaluate_fn = evaluate_fn

    def generate_samples(self) -> np.ndarray:
        if self.sampling_method == "lhs":
            return LHSSampler(seed=self.seed).sample(self.params, self.n_samples)
        elif self.sampling_method == "saltelli":
            return SaltelliSampler(seed=self.seed).sample(self.params, self.n_samples)
        raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

    def run(self, evaluate_fn: Callable[[np.ndarray], float] = None) -> "MonteCarloResult":
        from atha.monte_carlo.results import MonteCarloResult

        fn = evaluate_fn if evaluate_fn is not None else self._evaluate_fn
        if fn is None:
            raise ValueError("MonteCarloRunner.run requires evaluate_fn (constructor or run argument).")

        samples = self.generate_samples()

        def _safe_eval(X):
            try:
                val = fn(X)
                return float(val) if val is not None else float("nan")
            except Exception:
                return float("nan")

        if self.n_jobs == 1:
            Y = np.array([_safe_eval(X) for X in samples])
        else:
            # threading: avoids subprocess pickling of thermo backends (e.g. CoolProp)
            Y = np.array(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                    delayed(_safe_eval)(X) for X in samples
                )
            )

        converged = np.isfinite(Y)
        Y_valid = Y[converged]
        stats = compute_statistics(Y_valid) if len(Y_valid) > 0 else None

        return MonteCarloResult(
            param_names=[p.name for p in self.params],
            param_samples=samples,
            Y_samples=Y,
            converged=converged,
            stats=stats,
        )
