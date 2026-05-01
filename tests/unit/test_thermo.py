"""Unit tests for the thermodynamics layer."""
import math
import pytest
from atha.thermo.interface import FluidState, ThermoBackend
from atha.thermo.ideal_gas import IdealGasBackend

coolprop_available = pytest.mark.skipif(
    not __import__('importlib').util.find_spec('CoolProp'),
    reason="CoolProp not installed"
)
cantera_available = pytest.mark.skipif(
    not __import__('importlib').util.find_spec('cantera'),
    reason="Cantera not installed"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def air():
    return IdealGasBackend(gamma=1.4, R=287.0)


# ── FluidState ────────────────────────────────────────────────────────────────

def test_fluid_state_is_frozen():
    state = FluidState(P=1e5, T=300.0, h=4e5, rho=1.2, s=1000.0,
                       cp=1005.0, cv=718.0, gamma=1.4, mu=1.8e-5,
                       k=0.026, MW=0.029, phase='gas', quality=None)
    with pytest.raises((AttributeError, TypeError)):
        state.P = 2e5  # frozen dataclass must raise


def test_fluid_state_two_phase_quality():
    state = FluidState(P=5e5, T=151.8, h=2e5, rho=800.0, s=900.0,
                       cp=4000.0, cv=3000.0, gamma=1.33, mu=1e-4,
                       k=0.1, MW=0.018, phase='two-phase', quality=0.5)
    assert state.quality == 0.5


# ── ThermoBackend ─────────────────────────────────────────────────────────────

def test_thermo_backend_is_abstract():
    with pytest.raises(TypeError):
        ThermoBackend()


# ── IdealGasBackend ───────────────────────────────────────────────────────────

def test_ideal_gas_state_from_PT_density(air):
    state = air.state_from_PT(P=101325.0, T=300.0)
    expected_rho = 101325.0 / (287.0 * 300.0)
    assert abs(state.rho - expected_rho) < 1e-6

def test_ideal_gas_state_from_PT_gamma(air):
    state = air.state_from_PT(P=101325.0, T=300.0)
    assert abs(state.gamma - 1.4) < 1e-10

def test_ideal_gas_state_from_Ph_roundtrip(air):
    original = air.state_from_PT(P=5e5, T=800.0)
    recovered = air.state_from_Ph(P=original.P, h=original.h)
    assert abs(recovered.T - original.T) < 0.01
    assert abs(recovered.P - original.P) < 1.0

def test_ideal_gas_isentropic_temperature(air):
    inlet = air.state_from_PT(P=2e5, T=600.0)
    exit_state = air.isentropic_expansion(inlet, P_exit=1e5)
    expected_T = 600.0 * (1e5 / 2e5) ** ((1.4 - 1) / 1.4)
    assert abs(exit_state.T - expected_T) < 0.01

def test_ideal_gas_isentropic_entropy_conserved(air):
    inlet = air.state_from_PT(P=3e5, T=500.0)
    exit_state = air.isentropic_expansion(inlet, P_exit=1e5)
    assert abs(exit_state.s - inlet.s) < 1e-6   # entropy conserved

def test_ideal_gas_state_from_Ps_roundtrip(air):
    original = air.state_from_PT(P=2e5, T=400.0)
    recovered = air.state_from_Ps(P=original.P, s=original.s)
    assert abs(recovered.T - original.T) < 0.01

def test_ideal_gas_state_from_Ph_negative_enthalpy_raises(air):
    with pytest.raises(ValueError, match="h must be positive"):
        air.state_from_Ph(P=1e5, h=-100.0)

def test_ideal_gas_state_from_Ph_zero_enthalpy_raises(air):
    with pytest.raises(ValueError, match="h must be positive"):
        air.state_from_Ph(P=1e5, h=0.0)


# ── CoolProp (conditional — skip if not installed) ───────────────────────────

@coolprop_available
def test_coolprop_lox_liquid_phase():
    from atha.thermo.coolprop_backend import CoolPropBackend
    lox = CoolPropBackend("Oxygen")
    state = lox.state_from_PT(P=1e5, T=90.0)
    assert state.phase == 'liquid'

@coolprop_available
def test_coolprop_lox_density_at_nbp():
    from atha.thermo.coolprop_backend import CoolPropBackend
    lox = CoolPropBackend("Oxygen")
    # LOX normal boiling point: T_sat ≈ 90.188 K at P=1 atm → rho ≈ 1141 kg/m^3
    # Use T=90.18 K (just below T_sat) to ensure liquid phase
    state = lox.state_from_PT(P=101325.0, T=90.18)
    assert 1100.0 < state.rho < 1180.0, f"LOX density {state.rho:.1f} out of expected range"

@coolprop_available
def test_coolprop_ph_roundtrip():
    from atha.thermo.coolprop_backend import CoolPropBackend
    lox = CoolPropBackend("Oxygen")
    original = lox.state_from_PT(P=5e6, T=95.0)
    recovered = lox.state_from_Ph(P=original.P, h=original.h)
    assert abs(recovered.T - original.T) < 0.1

@coolprop_available
def test_coolprop_isentropic_entropy_conserved():
    from atha.thermo.coolprop_backend import CoolPropBackend
    # Use gaseous oxygen for a clean isentropic test
    ox_gas = CoolPropBackend("Oxygen")
    inlet = ox_gas.state_from_PT(P=2e6, T=400.0)
    exit_state = ox_gas.isentropic_expansion(inlet, P_exit=5e5)
    assert abs(exit_state.s - inlet.s) < 1.0   # J/(kg·K), loose tolerance for real gas


# ── Cantera (conditional — skip if not installed) ─────────────────────────────

@cantera_available
def test_cantera_lox_lh2_adiabatic_temperature():
    from atha.thermo.cantera_backend import CanteraBackend
    cb = CanteraBackend("h2o2.yaml")
    result = cb.combustion_equilibrium(
        fuel="H2", oxidizer="O2", MR=6.0, P_chamber=20e6
    )
    # LOX/LH2 at MR=6, 20MPa: T_ad should be in 3400–3800 K range
    assert 3400 < result.T_ad < 3800, f"T_ad={result.T_ad:.0f}K out of range"

@cantera_available
def test_cantera_c_star_reasonable():
    from atha.thermo.cantera_backend import CanteraBackend
    cb = CanteraBackend("h2o2.yaml")
    result = cb.combustion_equilibrium(
        fuel="H2", oxidizer="O2", MR=6.0, P_chamber=20e6
    )
    # LOX/LH2 c* is typically 2300–2450 m/s
    assert 2200 < result.c_star < 2500, f"c*={result.c_star:.0f} m/s out of range"
