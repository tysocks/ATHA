# atha/monte_carlo/profile_runner.py
from __future__ import annotations
from typing import Callable, List, Optional, Tuple

import numpy as np

from atha.monte_carlo.parameters import UncertainParameter
from atha.monte_carlo.runner import MonteCarloRunner
from atha.monte_carlo.results import MonteCarloResult


class ProfileMonteCarloRunner(MonteCarloRunner):
    """
    Monte Carlo over uncertain parameters where each sample may rebuild the
    engine/profile via ``apply_params_fn``, or reuse fixed ``layout``/``profile``.
    """

    def __init__(
        self,
        params: List[UncertainParameter],
        n_samples: int,
        layout=None,
        profile=None,
        X0: Optional[np.ndarray] = None,
        apply_params_fn: Optional[Callable[..., Tuple]] = None,
        extract_metric: Optional[Callable] = None,
        sampling_method: str = "lhs",
        sampler: Optional[str] = None,
        n_jobs: int = -1,
        seed: int = 42,
        verbose: int = 0,
    ):
        super().__init__(
            params=params,
            n_samples=n_samples,
            sampling_method=sampling_method,
            sampler=sampler,
            n_jobs=n_jobs,
            seed=seed,
            verbose=verbose,
        )
        self._layout = layout
        self._profile = profile
        self._X0 = X0
        self._apply_params_fn = apply_params_fn
        self._extract_metric = extract_metric or self._default_metric

    @staticmethod
    def _default_metric(profile_result) -> float:
        if not profile_result.success or not profile_result.phases:
            return float("nan")
        return float(profile_result.total_duration)

    def run(self, evaluate_fn=None) -> MonteCarloResult:
        from atha.profiles.limits import EngineAbort

        if evaluate_fn is not None:
            return super().run(evaluate_fn)

        if self._apply_params_fn is not None:
            names = [p.name for p in self.params]

            def evaluate_sample(Xrow: np.ndarray) -> float:
                d = {names[i]: float(Xrow[i]) for i in range(len(names))}
                try:
                    out = self._apply_params_fn(d)
                    if len(out) != 3:
                        raise TypeError(
                            "apply_params_fn must return (profile, layout, X0)"
                        )
                    profile, layout, X0 = out
                    profile_result = profile.execute(layout, X0)
                    return float(self._extract_metric(profile_result))
                except EngineAbort:
                    return float("nan")
                except Exception:
                    return float("nan")

            return super().run(evaluate_fn=evaluate_sample)

        profile = self._profile
        layout = self._layout
        X0 = self._X0
        if profile is None or layout is None or X0 is None:
            raise ValueError(
                "ProfileMonteCarloRunner requires apply_params_fn= "
                "or all of layout, profile, and X0."
            )

        def evaluate_with_profile(X: np.ndarray) -> float:
            del X  # fixed-topology run ignores sample row
            try:
                profile_result = profile.execute(layout, X0)
                return float(self._extract_metric(profile_result))
            except EngineAbort:
                return float("nan")
            except Exception:
                return float("nan")

        return super().run(evaluate_fn=evaluate_with_profile)
