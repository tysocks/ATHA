# tests/unit/test_monte_carlo.py
import numpy as np
import pytest
from atha.monte_carlo.parameters import UncertainParameter, ParameterType


def test_parameter_type_enum():
    assert ParameterType.NORMAL.value == "normal"
    assert ParameterType.UNIFORM.value == "uniform"
    assert ParameterType.LOGNORMAL.value == "lognormal"


def test_normal_parameter_samples_distribution():
    rng = np.random.default_rng(42)
    param = UncertainParameter(
        name="Pc", nominal=20.6e6,
        dist_type=ParameterType.NORMAL,
        sigma=0.4e6,
    )
    samples = np.array([param.sample(rng) for _ in range(2000)])
    assert abs(np.mean(samples) - 20.6e6) < 1e5     # mean within 0.5%
    assert abs(np.std(samples) - 0.4e6) < 5e4        # std within 12.5%


def test_uniform_parameter_samples_in_range():
    rng = np.random.default_rng(0)
    param = UncertainParameter(
        name="eta", nominal=0.975,
        dist_type=ParameterType.UNIFORM,
        lower=0.970, upper=0.990,
    )
    samples = np.array([param.sample(rng) for _ in range(500)])
    assert np.all(samples >= 0.970)
    assert np.all(samples <= 0.990)


def test_lognormal_parameter_positive():
    rng = np.random.default_rng(7)
    param = UncertainParameter(
        name="At", nominal=0.0687,
        dist_type=ParameterType.LOGNORMAL,
        sigma_log=0.01,
    )
    samples = np.array([param.sample(rng) for _ in range(200)])
    assert np.all(samples > 0), "Lognormal samples must be positive"
    assert abs(np.mean(samples) - 0.0687) < 0.002


# B2 Tests: LHSSampler and SaltelliSampler

from atha.monte_carlo.sampling import LHSSampler, SaltelliSampler


def _make_params():
    return [
        UncertainParameter("Pc", 20.6e6, ParameterType.NORMAL, sigma=0.4e6),
        UncertainParameter("MR", 6.0,    ParameterType.NORMAL, sigma=0.12),
        UncertainParameter("eta", 0.975, ParameterType.UNIFORM, lower=0.97, upper=0.99),
    ]


def test_lhs_sampler_shape():
    params = _make_params()
    sampler = LHSSampler(seed=42)
    samples = sampler.sample(params, N=100)
    assert samples.shape == (100, 3)


def test_lhs_sampler_covers_range():
    """LHS should cover tails for Normal param with N=200."""
    params = [UncertainParameter("x", 10.0, ParameterType.NORMAL, sigma=1.0)]
    sampler = LHSSampler(seed=0)
    samples = sampler.sample(params, N=200)
    assert samples[:, 0].min() < 8.0    # below mean - 2σ
    assert samples[:, 0].max() > 12.0   # above mean + 2σ


def test_saltelli_sampler_shape():
    """Saltelli generates N*(k+2) rows for k parameters."""
    params = _make_params()   # k=3
    sampler = SaltelliSampler(seed=99)
    samples = sampler.sample(params, N_base=50)
    assert samples.shape == (50 * (3 + 2), 3)   # 250 rows


def test_lhs_reproducible_with_seed():
    params = _make_params()
    s1 = LHSSampler(seed=5).sample(params, N=20)
    s2 = LHSSampler(seed=5).sample(params, N=20)
    np.testing.assert_array_equal(s1, s2)


# B3 Tests: MCStatistics

from atha.monte_carlo.statistics import MCStatistics, compute_statistics


def test_compute_statistics_known_distribution():
    rng = np.random.default_rng(123)
    Y = rng.normal(loc=450.0, scale=2.0, size=5000)
    stats = compute_statistics(Y)

    assert abs(stats.mean - 450.0) < 0.1
    assert abs(stats.std - 2.0) < 0.1
    assert stats.p5 < stats.median < stats.p95
    assert stats.N_samples == 5000
    assert stats.mean_ci_95 < 0.1


def test_compute_statistics_single_value():
    Y = np.array([100.0])
    stats = compute_statistics(Y)
    assert stats.mean == 100.0
    assert stats.std == 0.0


from atha.monte_carlo.runner import MonteCarloRunner


def test_mc_runner_serial_quadratic():
    """f(x) = x² with Normal x~N(2,0.1). E[x²] ≈ 4.01."""
    params = [UncertainParameter("x", nominal=2.0, dist_type=ParameterType.NORMAL, sigma=0.1)]
    runner = MonteCarloRunner(params=params, n_samples=500,
                               sampling_method="lhs", n_jobs=1, seed=42)
    result = runner.run(evaluate_fn=lambda X: X[0] ** 2)
    assert abs(result.stats.mean - 4.01) < 0.05
    assert result.stats.N_samples == 500
    assert len(result.Y_samples) == 500


def test_mc_runner_handles_nan_gracefully():
    """evaluate_fn returning NaN should be excluded from stats."""
    params = [UncertainParameter("x", nominal=0.0, dist_type=ParameterType.UNIFORM,
                                  lower=-1.0, upper=1.0)]
    runner = MonteCarloRunner(params=params, n_samples=200,
                               sampling_method="lhs", n_jobs=1, seed=0)
    result = runner.run(evaluate_fn=lambda X: float(np.sqrt(X[0])) if X[0] >= 0 else float("nan"))
    assert result.stats.N_samples < 200
    assert result.stats.N_samples > 50


def test_mc_runner_parallel_matches_serial():
    """n_jobs>1 should produce same mean as n_jobs=1 (same seed)."""
    params = [UncertainParameter("x", nominal=5.0, dist_type=ParameterType.NORMAL, sigma=1.0)]
    fn = lambda X: X[0] ** 2
    serial = MonteCarloRunner(params=params, n_samples=100, n_jobs=1, seed=7).run(fn)
    parallel = MonteCarloRunner(params=params, n_samples=100, n_jobs=2, seed=7).run(fn)
    assert abs(serial.stats.mean - parallel.stats.mean) < 0.5


from atha.monte_carlo.sensitivity import compute_sobol_indices
import math


def test_sobol_ishigami_function():
    """Ishigami function: x2 should have highest S1, x3 near zero."""
    params_ishi = [
        UncertainParameter("x1", 0.0, ParameterType.UNIFORM, lower=-math.pi, upper=math.pi),
        UncertainParameter("x2", 0.0, ParameterType.UNIFORM, lower=-math.pi, upper=math.pi),
        UncertainParameter("x3", 0.0, ParameterType.UNIFORM, lower=-math.pi, upper=math.pi),
    ]

    samples = SaltelliSampler(seed=42).sample(params_ishi, N_base=1000)

    def ishigami(X):
        return math.sin(X[0]) + 7 * math.sin(X[1])**2 + 0.1 * X[2]**4 * math.sin(X[0])

    Y = np.array([ishigami(X) for X in samples])
    Si = compute_sobol_indices(params_ishi, samples, Y, N_base=1000)

    assert Si["S1"][1] > Si["S1"][0], "x2 should dominate over x1 in first-order"
    assert Si["S1"][2] < 0.05, "x3 should have near-zero first-order index"
    assert np.all(Si["S1"] >= -0.1)
    assert np.all(Si["ST"] >= -0.1)
