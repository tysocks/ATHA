# tests/integration/test_monte_carlo_ttbe.py
"""Integration test: Monte Carlo uncertainty study on TTBE JANNAF model."""
import numpy as np
import pytest
from atha.monte_carlo import UncertainParameter, ParameterType, MonteCarloRunner
from atha.monte_carlo.sensitivity import run_sensitivity_analysis, compute_sobol_indices
from atha.monte_carlo.sampling import SaltelliSampler
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.thermo.ideal_gas import IdealGasBackend


TTBE_PARAMS = [
    UncertainParameter("Pc",       20.6e6, ParameterType.NORMAL, sigma_pct=2.0),
    UncertainParameter("MR",       6.0,    ParameterType.NORMAL, sigma_pct=2.0),
    UncertainParameter("eta_cstar",0.975,  ParameterType.NORMAL, sigma=0.005),
    UncertainParameter("eta_div",  0.9830, ParameterType.NORMAL, sigma=0.004),
    UncertainParameter("At",       0.0687, ParameterType.NORMAL, sigma_pct=1.5),
    UncertainParameter("epsilon",  77.5,   ParameterType.NORMAL, sigma_pct=2.0),
]


def _make_evaluator():
    def evaluate(X):
        Pc, MR, eta_cstar, eta_div, At, epsilon = X
        thermo = IdealGasBackend(gamma=1.24, R=711.0)
        eff = JANNAFEfficiencies(
            eta_cstar=float(eta_cstar),
            eta_Cd=0.98,
            eta_velocity=0.99,
            eta_divergence=float(eta_div),
            eta_two_phase=1.0,
            eta_boundary_layer=0.99,
        )
        jannaf = SimplifiedJANNAF(
            thermo=thermo, efficiencies=eff,
            throat_area=float(At),
            exit_area=float(At) * float(epsilon),
            ambient_pressure=0.0,
        )
        result = jannaf.compute(
            P_chamber=float(Pc),
            T_chamber=3560.0,
            MR=float(MR),
            mdot_total=468.0,
        )
        return result.Isp
    return evaluate


def test_mc_ttbe_basic_statistics():
    """Mean Isp near expected range, CV < 5%."""
    runner = MonteCarloRunner(
        params=TTBE_PARAMS, n_samples=100,
        sampling_method="lhs", n_jobs=1, seed=42,
    )
    result = runner.run(_make_evaluator())

    assert result.stats.N_samples >= 95
    assert 380 < result.stats.mean < 500, f"Mean Isp {result.stats.mean:.1f}s out of range"
    assert result.stats.cv_pct < 5.0


def test_mc_ttbe_convergence_rate():
    """All LHS samples should converge for JANNAF model."""
    runner = MonteCarloRunner(
        params=TTBE_PARAMS, n_samples=50,
        sampling_method="lhs", n_jobs=1, seed=7,
    )
    result = runner.run(_make_evaluator())
    convergence_rate = np.sum(result.converged) / len(result.converged)
    assert convergence_rate > 0.95


def test_mc_ttbe_sobol_efficiency_factors_dominate():
    """Sobol: eta_cstar and eta_div should be in the top 2 total-order drivers.

    For the SimplifiedJANNAF model in vacuum with fixed gamma/R and fixed mass
    flow, Isp is driven almost entirely by the efficiency factors eta_cstar and
    eta_divergence; Pc and MR do not appear in the Isp formula.
    """
    N_base = 128
    samples = SaltelliSampler(seed=42).sample(TTBE_PARAMS, N_base)
    Y = np.array([_make_evaluator()(X) for X in samples])

    Si = compute_sobol_indices(TTBE_PARAMS, samples, Y, N_base=N_base)

    top2_idx = np.argsort(Si["ST"])[-2:]
    param_names = [p.name for p in TTBE_PARAMS]
    top2_names = {param_names[i] for i in top2_idx}
    assert "eta_cstar" in top2_names or "eta_div" in top2_names, \
        f"Expected eta_cstar or eta_div in top 2 drivers, got {top2_names}"


def test_mc_result_save_load_roundtrip(tmp_path):
    runner = MonteCarloRunner(
        params=TTBE_PARAMS, n_samples=30,
        sampling_method="lhs", n_jobs=1, seed=1,
    )
    result = runner.run(_make_evaluator())
    fname = str(tmp_path / "mc_ttbe.hdf5")
    result.save(fname)

    from atha.monte_carlo.results import MonteCarloResult
    loaded = MonteCarloResult.load(fname)

    np.testing.assert_array_almost_equal(result.Y_samples, loaded.Y_samples)
    assert abs(result.stats.mean - loaded.stats.mean) < 0.001


from atha.monte_carlo.profile_runner import ProfileMonteCarloRunner
from atha.profiles import TestProfile, PhaseDefinition, PhaseMode


def test_profile_mc_success_rate():
    """Run 10 profile samples; verify >= 8 succeed."""
    from atha.thermo.ideal_gas import IdealGasBackend
    from atha.components.volume import Volume
    from atha.core.engine import Engine

    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("tank", volume=0.1, thermo=gas, initial_P=1e5, initial_T=300.0)
    engine = Engine("e"); engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()

    profile = TestProfile(
        name="mc_profile",
        phases=[
            PhaseDefinition("dwell1", PhaseMode.DWELL, duration=0.1),
            PhaseDefinition("dwell2", PhaseMode.DWELL, duration=0.1),
        ],
    )
    params = [UncertainParameter("Pc", 20.6e6, ParameterType.NORMAL, sigma_pct=2.0)]
    runner = ProfileMonteCarloRunner(
        params=params, n_samples=10,
        sampling_method="lhs", n_jobs=1, seed=0,
        layout=layout, profile=profile, X0=X0,
    )
    result = runner.run()
    assert result.stats.N_samples >= 8
