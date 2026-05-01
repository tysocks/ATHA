# tests/unit/test_components.py
"""Unit tests for engine components."""
import math
import pytest
import numpy as np
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume


@pytest.fixture
def air():
    return IdealGasBackend(gamma=1.4, R=287.0)


@pytest.fixture
def small_volume(air):
    vol = Volume("vol1", volume=0.01, thermo=air, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    vol.add_outlet("outlet")
    return vol


# ── Volume construction ───────────────────────────────────────────────────────

def test_volume_has_two_states(small_volume):
    assert small_volume.n_states == 2
    assert small_volume.state_names == ["P", "h"]

def test_volume_initial_pressure(small_volume):
    assert small_volume._state_values["P"] == 1e5

def test_volume_initial_enthalpy_from_temperature(air):
    vol = Volume("v", volume=0.01, thermo=air, initial_P=1e5, initial_T=500.0)
    vol.add_inlet()
    expected_h = air.cp * 500.0   # h = Cp * T for ideal gas
    assert abs(vol._state_values["h"] - expected_h) < 1.0

def test_volume_ports_registered(small_volume):
    assert "inlet" in small_volume.ports
    assert "outlet" in small_volume.ports

def test_volume_no_algebraic_vars(small_volume):
    assert small_volume.n_algebraic == 0


# ── Volume: compute_outputs ───────────────────────────────────────────────────

def test_volume_compute_outputs_returns_fluid_state(small_volume, air):
    states = {"P": 1e5, "h": air.cp * 300.0}
    outputs = small_volume.compute_outputs(0.0, states, {})
    assert "fluid_state" in outputs
    assert "T" in outputs
    assert "rho" in outputs
    # T should match h/cp for ideal gas
    assert abs(outputs["T"] - 300.0) < 0.1


# ── Volume: get_state_derivatives ────────────────────────────────────────────

def test_volume_dPdt_positive_for_net_inflow(small_volume, air):
    states = {"P": 1e5, "h": air.cp * 300.0}
    inputs = {"inlet.mdot": 1.0, "inlet.h": air.cp * 300.0,
              "outlet.mdot": 0.0}
    outputs = small_volume.compute_outputs(0.0, states, inputs)
    derivs = small_volume.get_state_derivatives(0.0, states, inputs, outputs)
    assert derivs["P"] > 0.0, "Pressure should rise with net inflow"

def test_volume_dPdt_zero_for_equal_flow(small_volume, air):
    states = {"P": 1e5, "h": air.cp * 300.0}
    inputs = {"inlet.mdot": 1.0, "inlet.h": air.cp * 300.0,
              "outlet.mdot": 1.0}
    outputs = small_volume.compute_outputs(0.0, states, inputs)
    derivs = small_volume.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["P"]) < 1.0, "Pressure should be steady with balanced flow"

def test_volume_dPdt_negative_for_net_outflow(small_volume, air):
    states = {"P": 1e5, "h": air.cp * 300.0}
    inputs = {"inlet.mdot": 0.0,
              "outlet.mdot": 1.0}
    outputs = small_volume.compute_outputs(0.0, states, inputs)
    derivs = small_volume.get_state_derivatives(0.0, states, inputs, outputs)
    assert derivs["P"] < 0.0, "Pressure should drop with net outflow"

def test_volume_dPdt_magnitude(air):
    """Verify dP/dt against analytical formula: dP/dt = gamma*P/(rho*V) * mdot_net/rho"""
    V = 0.01  # m^3
    P0, T0 = 1e5, 300.0
    vol = Volume("v", volume=V, thermo=air, initial_P=P0, initial_T=T0)
    vol.add_inlet()
    fs0 = air.state_from_PT(P0, T0)
    states = {"P": P0, "h": fs0.h}
    mdot_in = 0.5  # kg/s
    inputs = {"inlet.mdot": mdot_in, "inlet.h": fs0.h}
    outputs = vol.compute_outputs(0.0, states, inputs)
    derivs = vol.get_state_derivatives(0.0, states, inputs, outputs)
    # Analytical: dP/dt = gamma * R * T / V * (mdot_net / rho)
    # For ideal gas: gamma*R*T = gamma*P/rho, so dP/dt = gamma*P/(rho*V)*mdot_net/rho
    # Actually: dP/dt = gamma * R_eff * T / V * mdot_net / rho
    # = gamma * (cp-cv) * T / V * mdot_in / rho
    R_eff = air.cp - air.cv
    expected_dPdt = (air.gamma * R_eff * T0 / V) * (mdot_in / fs0.rho)
    assert abs(derivs["P"] - expected_dPdt) / expected_dPdt < 0.001

def test_volume_heat_input_increases_dh(air):
    V = 0.01
    vol = Volume("v", volume=V, thermo=air, initial_P=1e5, initial_T=300.0)
    vol.add_inlet()
    vol.add_thermal_port("heat")
    fs0 = air.state_from_PT(1e5, 300.0)
    states = {"P": 1e5, "h": fs0.h}
    # No flow, just heat
    inputs_no_heat = {"inlet.mdot": 0.0}
    inputs_with_heat = {"inlet.mdot": 0.0, "heat.Q_dot": 1000.0}
    outputs = vol.compute_outputs(0.0, states, inputs_no_heat)
    derivs_no_heat = vol.get_state_derivatives(0.0, states, inputs_no_heat, outputs)
    derivs_with_heat = vol.get_state_derivatives(0.0, states, inputs_with_heat, outputs)
    assert derivs_with_heat["h"] > derivs_no_heat["h"], "Heat input should increase dh/dt"


# ── Volume: initialize ────────────────────────────────────────────────────────

def test_volume_initialize_sets_states(air):
    vol = Volume("v", volume=0.01, thermo=air)
    vol.add_inlet()
    vol.initialize({"P": 2e5, "T": 500.0})
    assert abs(vol._state_values["P"] - 2e5) < 1.0
    assert abs(vol._state_values["h"] - air.cp * 500.0) < 1.0
