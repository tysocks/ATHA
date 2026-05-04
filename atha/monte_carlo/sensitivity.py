# atha/monte_carlo/sensitivity.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
from atha.monte_carlo.parameters import UncertainParameter


def compute_sobol_indices(
    params: List[UncertainParameter],
    param_samples: np.ndarray,
    Y: np.ndarray,
    N_base: int,
    calc_second_order: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute first-order and total-order Sobol sensitivity indices via SALib."""
    from SALib.analyze import sobol

    k = len(params)
    problem = {
        "num_vars": k,
        "names": [p.name for p in params],
        "bounds": [[0.0, 1.0] for _ in range(k)],
    }

    Si = sobol.analyze(
        problem, Y,
        calc_second_order=calc_second_order,
        conf_level=0.95,
        print_to_console=False,
    )
    return {
        "S1":      np.array(Si["S1"]),
        "ST":      np.array(Si["ST"]),
        "S1_conf": np.array(Si["S1_conf"]),
        "ST_conf": np.array(Si["ST_conf"]),
    }


def run_sensitivity_analysis(
    params: List[UncertainParameter],
    evaluate_fn,
    N_base: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[str, np.ndarray]:
    """Convenience: sample + evaluate + compute Sobol indices."""
    from atha.monte_carlo.sampling import SaltelliSampler
    from joblib import Parallel, delayed

    samples = SaltelliSampler(seed=seed).sample(params, N_base)

    def _safe(X):
        try:
            return float(evaluate_fn(X))
        except Exception:
            return float("nan")

    if n_jobs == 1:
        Y = np.array([_safe(X) for X in samples])
    else:
        Y = np.array(
            Parallel(n_jobs=n_jobs, backend="loky")(delayed(_safe)(X) for X in samples)
        )

    return compute_sobol_indices(params, samples, Y, N_base=N_base)
