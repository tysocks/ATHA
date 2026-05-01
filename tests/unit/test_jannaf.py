# tests/unit/test_jannaf.py
"""Unit tests for JANNAF simplified performance module."""
import math
import pytest
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.jannaf.performance import PerformanceResult
from atha.jannaf.simplified import SimplifiedJANNAF, G0
from atha.thermo.ideal_gas import IdealGasBackend


@pytest.fixture
def air_thermo():
    return IdealGasBackend(gamma=1.4, R=287.0)


@pytest.fixture
def lox_lh2_approx():
    """Approximate LOX/LH2 combustion products: gamma=1.2, R=520 J/(kg·K)."""
    return IdealGasBackend(gamma=1.2, R=520.0)


@pytest.fixture
def default_eff():
    return JANNAFEfficiencies()


# ── JANNAFEfficiencies ────────────────────────────────────────────────────────

def test_efficiencies_default_divergence():
    eff = JANNAFEfficiencies()
    expected = 0.5 * (1.0 + math.cos(math.radians(15.0)))
    assert abs(eff.eta_divergence - expected) < 1e-10

def test_efficiencies_default_values_in_range():
    eff = JANNAFEfficiencies()
    for attr in ["eta_cstar", "eta_Cd", "eta_velocity",
                 "eta_divergence", "eta_two_phase", "eta_boundary_layer"]:
        v = getattr(eff, attr)
        assert 0.0 < v <= 1.0, f"{attr}={v} out of (0, 1]"

def test_efficiencies_invalid_raises():
    with pytest.raises(ValueError):
        JANNAFEfficiencies(eta_cstar=1.5)   # > 1 not allowed

def test_efficiencies_overall_Cf():
    eff = JANNAFEfficiencies(eta_velocity=0.99, eta_divergence=0.98, eta_boundary_layer=0.99)
    expected = 0.99 * 0.98 * 0.99
    assert abs(eff.overall_Cf_efficiency - expected) < 1e-12


# ── SimplifiedJANNAF._exit_mach ──────────────────────────────────────────────

def test_exit_mach_epsilon_1_is_1():
    # At epsilon=1 (throat), M=1
    Me = SimplifiedJANNAF._exit_mach(gamma=1.4, epsilon=1.0)
    assert abs(Me - 1.0) < 0.01

def test_exit_mach_large_epsilon():
    # For gamma=1.4, epsilon=77: Me should be around 5-6
    Me = SimplifiedJANNAF._exit_mach(gamma=1.4, epsilon=77.0)
    assert 4.0 < Me < 8.0

def test_exit_mach_gamma_effect():
    # Higher gamma → higher exit Mach for same area ratio
    # (lower gamma produces larger area ratios at same M, so needs lower M for same epsilon)
    Me_14 = SimplifiedJANNAF._exit_mach(gamma=1.4, epsilon=10.0)
    Me_12 = SimplifiedJANNAF._exit_mach(gamma=1.2, epsilon=10.0)
    assert Me_14 > Me_12


# ── SimplifiedJANNAF._ideal_cf ────────────────────────────────────────────────

def test_ideal_cf_vacuum_greater_than_sealevel():
    """Vacuum Cf (Pa=0) > sea-level Cf (Pa=101325) for same expansion."""
    Cf_vac = SimplifiedJANNAF._ideal_cf(1.4, 20e6, 1e4, 0.0,     77.0)
    Cf_sl  = SimplifiedJANNAF._ideal_cf(1.4, 20e6, 1e4, 101325.0, 77.0)
    assert Cf_vac > Cf_sl

def test_ideal_cf_positive():
    Cf = SimplifiedJANNAF._ideal_cf(1.4, 10e6, 5e4, 0.0, 20.0)
    assert Cf > 0.0


# ── SimplifiedJANNAF.compute ──────────────────────────────────────────────────

def test_jannaf_compute_returns_performance_result(lox_lh2_approx, default_eff):
    jannaf = SimplifiedJANNAF(
        thermo=lox_lh2_approx, efficiencies=default_eff,
        throat_area=0.05, exit_area=3.85, ambient_pressure=0.0
    )
    result = jannaf.compute(P_chamber=20e6, T_chamber=3600.0, MR=6.0, mdot_total=500.0)
    assert isinstance(result, PerformanceResult)

def test_jannaf_isp_in_expected_range_lox_lh2(lox_lh2_approx, default_eff):
    """LOX/LH2 at Pc=20MPa, Ae/At=77, vacuum: Isp should be 400-500s."""
    jannaf = SimplifiedJANNAF(
        thermo=lox_lh2_approx, efficiencies=default_eff,
        throat_area=0.0687, exit_area=5.29, ambient_pressure=0.0
    )
    result = jannaf.compute(P_chamber=20e6, T_chamber=3600.0, MR=6.0, mdot_total=500.0)
    assert 380 < result.Isp < 520, f"Isp={result.Isp:.1f}s outside expected range"

def test_jannaf_thrust_consistent_with_isp(lox_lh2_approx, default_eff):
    """Thrust = Cf * Pc * At, Isp = c_star * Cf / g0; consistent via c* = Pc*At/mdot_throat."""
    jannaf = SimplifiedJANNAF(
        thermo=lox_lh2_approx, efficiencies=default_eff,
        throat_area=0.0687, exit_area=5.29, ambient_pressure=0.0
    )
    result = jannaf.compute(P_chamber=20e6, T_chamber=3600.0, MR=6.0, mdot_total=500.0)
    # Implied throat mass flow from c* definition: mdot_throat = Pc * At / c*
    mdot_throat = result.P_chamber * jannaf.A_t / result.c_star
    thrust_from_isp = mdot_throat * result.Isp * G0
    assert abs(result.thrust - thrust_from_isp) / result.thrust < 0.01

def test_jannaf_delivered_less_than_ideal(lox_lh2_approx, default_eff):
    """Delivered Isp < ideal Isp due to efficiency factors < 1."""
    jannaf = SimplifiedJANNAF(
        thermo=lox_lh2_approx, efficiencies=default_eff,
        throat_area=0.05, exit_area=3.85, ambient_pressure=0.0
    )
    result = jannaf.compute(P_chamber=15e6, T_chamber=3500.0, MR=6.0, mdot_total=200.0)
    assert result.Isp < result.Isp_ideal
    assert result.c_star < result.c_star_ideal

def test_jannaf_perfect_efficiency_equals_ideal(lox_lh2_approx):
    """With all efficiencies=1.0, delivered == ideal."""
    perfect = JANNAFEfficiencies(
        eta_cstar=1.0, eta_Cd=1.0, eta_velocity=1.0,
        eta_divergence=1.0, eta_two_phase=1.0, eta_boundary_layer=1.0,
    )
    jannaf = SimplifiedJANNAF(
        thermo=lox_lh2_approx, efficiencies=perfect,
        throat_area=0.05, exit_area=3.85, ambient_pressure=0.0
    )
    result = jannaf.compute(P_chamber=15e6, T_chamber=3500.0, MR=6.0, mdot_total=200.0)
    assert abs(result.c_star - result.c_star_ideal) < 1.0
    assert abs(result.Isp - result.Isp_ideal) < 0.5

def test_jannaf_higher_Pc_higher_Isp(lox_lh2_approx, default_eff):
    """Higher chamber pressure → higher Isp at sea level (smaller -Pa/Pc*eps penalty)."""
    P_a = 101325.0
    jannaf_lo = SimplifiedJANNAF(lox_lh2_approx, default_eff, 0.05, 3.85, P_a)
    jannaf_hi = SimplifiedJANNAF(lox_lh2_approx, default_eff, 0.05, 3.85, P_a)
    r_lo = jannaf_lo.compute(P_chamber=5e6,  T_chamber=3500.0, MR=6.0, mdot_total=100.0)
    r_hi = jannaf_hi.compute(P_chamber=20e6, T_chamber=3500.0, MR=6.0, mdot_total=100.0)
    assert r_hi.Isp > r_lo.Isp
