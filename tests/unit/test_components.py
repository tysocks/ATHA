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


# ── Rotor tests ───────────────────────────────────────────────────────────────

from atha.components.rotor import Rotor


def test_rotor_has_one_state():
    rotor = Rotor("r1", inertia=10.0)
    assert rotor.n_states == 1
    assert rotor.state_names == ["omega"]


def test_rotor_accelerates_under_net_torque():
    rotor = Rotor("r1", inertia=10.0, friction_coeff=0.0)
    states = {"omega": 0.0}
    inputs = {"shaft_in.tau": 100.0, "shaft_out.tau": 0.0}
    outputs = rotor.compute_outputs(0.0, states, inputs)
    derivs = rotor.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["omega"] - 10.0) < 1e-10  # 100/10 = 10 rad/s^2


def test_rotor_friction_reduces_acceleration():
    rotor = Rotor("r1", inertia=10.0, friction_coeff=1.0)
    states = {"omega": 5.0}  # 1.0 * 5 = 5 N·m friction
    inputs = {"shaft_in.tau": 100.0, "shaft_out.tau": 0.0}
    outputs = rotor.compute_outputs(0.0, states, inputs)
    derivs = rotor.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["omega"] - 9.5) < 1e-10  # (100 - 5) / 10 = 9.5


def test_rotor_equilibrium_omega():
    # At steady state: tau_drive = tau_load + k_f * omega -> omega = (drive-load)/k_f
    rotor = Rotor("r1", inertia=5.0, friction_coeff=2.0)
    states = {"omega": 50.0}  # (100 - 0) / 2 = 50 rad/s
    inputs = {"shaft_in.tau": 100.0, "shaft_out.tau": 0.0}
    outputs = rotor.compute_outputs(0.0, states, inputs)
    derivs = rotor.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["omega"]) < 1e-10


def test_rotor_initialize():
    rotor = Rotor("r1", inertia=1.0, initial_omega=314.16)
    assert abs(rotor._state_values["omega"] - 314.16) < 1e-10


# ── PipeWithInertia tests ─────────────────────────────────────────────────────

from atha.components.pipe_inertia import PipeWithInertia


def test_pipe_inertia_has_one_state():
    pipe = PipeWithInertia("p1", length=1.0, diameter=0.05)
    assert pipe.n_states == 1
    assert pipe.state_names == ["mdot"]


def test_pipe_inertia_positive_dmdot_for_pressure_diff():
    pipe = PipeWithInertia("p1", length=1.0, diameter=0.05, friction_factor=0.0)
    A = math.pi * (0.025) ** 2
    states = {"mdot": 0.0}
    inputs = {"inlet.P": 1e5 + 1000.0, "outlet.P": 1e5, "inlet.rho": 1.0}
    outputs = pipe.compute_outputs(0.0, states, inputs)
    derivs = pipe.get_state_derivatives(0.0, states, inputs, outputs)
    expected = (A / 1.0) * 1000.0
    assert abs(derivs["mdot"] - expected) / expected < 1e-6


def test_pipe_inertia_friction_reduces_dmdot():
    pipe = PipeWithInertia("p1", length=1.0, diameter=0.1, friction_factor=0.02)
    A = math.pi * 0.05 ** 2
    states = {"mdot": 1.0}  # 1 kg/s
    inputs = {"inlet.P": 2e5, "outlet.P": 1e5, "inlet.rho": 1.0}
    outputs = pipe.compute_outputs(0.0, states, inputs)
    derivs_no_f = PipeWithInertia("p2", length=1.0, diameter=0.1, friction_factor=0.0)
    outputs2 = derivs_no_f.compute_outputs(0.0, states, inputs)
    derivs2 = derivs_no_f.get_state_derivatives(0.0, states, inputs, outputs2)
    derivs = pipe.get_state_derivatives(0.0, states, inputs, outputs)
    assert derivs["mdot"] < derivs2["mdot"]


def test_pipe_inertia_initialize():
    pipe = PipeWithInertia("p1", length=1.0, diameter=0.05, initial_mdot=2.5)
    assert abs(pipe._state_values["mdot"] - 2.5) < 1e-10


# ── PipeAlgebraic tests ───────────────────────────────────────────────────────

from atha.components.pipe_algebraic import PipeAlgebraic


def test_pipe_algebraic_no_states():
    pipe = PipeAlgebraic("p1", length=1.0, diameter=0.05)
    assert pipe.n_states == 0


def test_pipe_algebraic_flow_from_pressure_drop():
    pipe = PipeAlgebraic("p1", length=1.0, diameter=0.1, friction_factor=0.02)
    A = math.pi * 0.05 ** 2
    K = 0.02 * 1.0 / 0.1  # f * L/D = 0.2
    rho = 1000.0  # water
    dP = 1e4  # 10 kPa
    v_expected = math.sqrt(2 * dP / (K * rho))
    mdot_expected = rho * A * v_expected
    inputs = {"inlet.P": 1e5 + dP, "outlet.P": 1e5, "inlet.rho": rho}
    outputs = pipe.compute_outputs(0.0, {}, inputs)
    assert abs(outputs["mdot"] - mdot_expected) / mdot_expected < 1e-6


def test_pipe_algebraic_zero_dP_zero_flow():
    pipe = PipeAlgebraic("p1", length=1.0, diameter=0.05)
    inputs = {"inlet.P": 1e5, "outlet.P": 1e5, "inlet.rho": 1.0}
    outputs = pipe.compute_outputs(0.0, {}, inputs)
    assert abs(outputs["mdot"]) < 1e-12


def test_pipe_algebraic_reverse_flow():
    pipe = PipeAlgebraic("p1", length=1.0, diameter=0.1, friction_factor=0.02)
    inputs = {"inlet.P": 1e5, "outlet.P": 1.5e5, "inlet.rho": 1000.0}
    outputs = pipe.compute_outputs(0.0, {}, inputs)
    assert outputs["mdot"] < 0.0  # reverse flow


# ── OrificeCompressible tests ─────────────────────────────────────────────────

from atha.components.orifice import OrificeCompressible


def test_orifice_no_states():
    orifice = OrificeCompressible("o1", area=1e-4)
    assert orifice.n_states == 0


def test_orifice_choked_flow():
    # High pressure ratio → choked
    orifice = OrificeCompressible("o1", area=1e-4, discharge_coeff=1.0,
                                   gamma=1.4, R_gas=287.0)
    inputs = {"inlet.P": 2e5, "outlet.P": 1e5, "inlet.T": 300.0}
    outputs = orifice.compute_outputs(0.0, {}, inputs)
    gamma, R = 1.4, 287.0
    choked_coeff = (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    mdot_choked = 1e-4 * 2e5 * math.sqrt(gamma / (R * 300.0)) * choked_coeff
    assert abs(outputs["mdot"] - mdot_choked) / mdot_choked < 0.02  # within 2% (blending)


def test_orifice_zero_flow_for_reverse_pressure():
    orifice = OrificeCompressible("o1", area=1e-4)
    inputs = {"inlet.P": 1e5, "outlet.P": 2e5, "inlet.T": 300.0}
    outputs = orifice.compute_outputs(0.0, {}, inputs)
    assert outputs["mdot"] == 0.0


def test_orifice_mdot_positive():
    orifice = OrificeCompressible("o1", area=1e-4)
    inputs = {"inlet.P": 1.1e5, "outlet.P": 1.0e5, "inlet.T": 300.0}
    outputs = orifice.compute_outputs(0.0, {}, inputs)
    assert outputs["mdot"] > 0.0


def test_orifice_choked_exceeds_subsonic():
    # At high PR, choked flow > theoretical subsonic continuation
    orifice = OrificeCompressible("o1", area=1e-4, discharge_coeff=1.0)
    inputs_hi = {"inlet.P": 10e5, "outlet.P": 0.1e5, "inlet.T": 300.0}
    inputs_lo = {"inlet.P": 1.05e5, "outlet.P": 1.0e5, "inlet.T": 300.0}
    out_hi = orifice.compute_outputs(0.0, {}, inputs_hi)
    out_lo = orifice.compute_outputs(0.0, {}, inputs_lo)
    # High PR should give more flow than low PR
    assert out_hi["mdot"] > out_lo["mdot"]
