# atha/monte_carlo/statistics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MCStatistics:
    N_samples: int
    mean: float
    std: float
    cv_pct: float
    min: float
    max: float
    median: float
    p1: float
    p5: float
    p95: float
    p99: float
    mean_ci_95: float


def compute_statistics(Y: np.ndarray) -> MCStatistics:
    Y = np.asarray(Y, dtype=float)
    Y_valid = Y[np.isfinite(Y)]
    N = len(Y_valid)

    if N == 0:
        raise ValueError("No finite samples to compute statistics from")

    mu = float(np.mean(Y_valid))
    sigma = float(np.std(Y_valid, ddof=min(1, N - 1)))

    return MCStatistics(
        N_samples=N,
        mean=mu,
        std=sigma,
        cv_pct=100.0 * sigma / abs(mu) if mu != 0 else float("inf"),
        min=float(np.min(Y_valid)),
        max=float(np.max(Y_valid)),
        median=float(np.median(Y_valid)),
        p1=float(np.percentile(Y_valid, 1)),
        p5=float(np.percentile(Y_valid, 5)),
        p95=float(np.percentile(Y_valid, 95)),
        p99=float(np.percentile(Y_valid, 99)),
        mean_ci_95=1.96 * sigma / np.sqrt(N) if N > 1 else 0.0,
    )
