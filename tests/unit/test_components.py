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
    rotor = Rotor("r1", moment_of_inertia=10.0)
    assert rotor.n_states == 1
    assert rotor.state_names == ["omega"]


def test_rotor_accelerates_under_net_torque():
    rotor = Rotor("r1", moment_of_inertia=10.0, friction_coeff=0.0)
    rotor.port("turbine_in")   # INLET drive port (name contains "turbine")
    rotor.port("pump_out")     # OUTLET load port
    states = {"omega": 0.0}
    inputs = {"turbine_in.tau": 100.0, "pump_out.tau": 0.0}
    outputs = rotor.compute_outputs(0.0, states, inputs)
    derivs = rotor.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["omega"] - 10.0) < 1e-10  # 100/10 = 10 rad/s^2


def test_rotor_friction_reduces_acceleration():
    rotor = Rotor("r1", moment_of_inertia=10.0, friction_coeff=1.0)
    rotor.port("turbine_in")
    rotor.port("pump_out")
    states = {"omega": 5.0}  # friction = 1.0 * 5 = 5 N·m
    inputs = {"turbine_in.tau": 100.0, "pump_out.tau": 0.0}
    outputs = rotor.compute_outputs(0.0, states, inputs)
    derivs = rotor.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["omega"] - 9.5) < 1e-10  # (100 - 5) / 10 = 9.5


def test_rotor_equilibrium_omega():
    # At steady state: tau_drive = tau_load + k_f * omega -> omega = (drive-load)/k_f
    rotor = Rotor("r1", moment_of_inertia=5.0, friction_coeff=2.0)
    rotor.port("turbine_in")
    rotor.port("pump_out")
    states = {"omega": 50.0}  # (100 - 0) / 2 = 50 rad/s
    inputs = {"turbine_in.tau": 100.0, "pump_out.tau": 0.0}
    outputs = rotor.compute_outputs(0.0, states, inputs)
    derivs = rotor.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["omega"]) < 1e-10


def test_rotor_initialize():
    rotor = Rotor("r1", moment_of_inertia=1.0)
    rotor.initialize({"omega": 314.16})
    assert abs(rotor._state_values["omega"] - 314.16) < 1e-10


def test_rotor_legacy_inertia_kwarg():
    # Backward compatibility: inertia= alias for moment_of_inertia=
    rotor = Rotor("r1", inertia=10.0)
    assert rotor.n_states == 1


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


# ── Valve tests ───────────────────────────────────────────────────────────────

from atha.components.valve import Valve


def test_valve_no_states():
    v = Valve("v1", max_area=1e-3)
    assert v.n_states == 0


def test_valve_fully_open():
    v = Valve("v1", max_area=1e-3, discharge_coeff=1.0)
    rho = 1000.0
    dP = 1e4
    expected = 1e-3 * math.sqrt(2 * rho * dP)
    inputs = {"inlet.P": 1e5 + dP, "outlet.P": 1e5, "inlet.rho": rho, "valve.A_frac": 1.0}
    out = v.compute_outputs(0.0, {}, inputs)
    assert abs(out["mdot"] - expected) / expected < 1e-6


def test_valve_half_open_halves_flow():
    v = Valve("v1", max_area=1e-3, discharge_coeff=1.0)
    rho, dP = 1000.0, 1e4
    inputs_full = {"inlet.P": 1e5+dP, "outlet.P": 1e5, "inlet.rho": rho, "valve.A_frac": 1.0}
    inputs_half = {"inlet.P": 1e5+dP, "outlet.P": 1e5, "inlet.rho": rho, "valve.A_frac": 0.5}
    out_full = v.compute_outputs(0.0, {}, inputs_full)
    out_half = v.compute_outputs(0.0, {}, inputs_half)
    assert abs(out_half["mdot"] - 0.5 * out_full["mdot"]) / out_full["mdot"] < 1e-6


def test_valve_closed_zero_flow():
    v = Valve("v1", max_area=1e-3)
    inputs = {"inlet.P": 2e5, "outlet.P": 1e5, "inlet.rho": 1000.0, "valve.A_frac": 0.0}
    out = v.compute_outputs(0.0, {}, inputs)
    assert out["mdot"] == 0.0


def test_valve_reverse_flow():
    v = Valve("v1", max_area=1e-3, discharge_coeff=1.0)
    inputs = {"inlet.P": 1e5, "outlet.P": 2e5, "inlet.rho": 1000.0, "valve.A_frac": 1.0}
    out = v.compute_outputs(0.0, {}, inputs)
    assert out["mdot"] < 0.0


# ── MetalNode tests ───────────────────────────────────────────────────────────

from atha.components.metal_node import MetalNode


def test_metal_node_has_one_state():
    mn = MetalNode("wall", mass=1.0)
    assert mn.n_states == 1
    assert mn.state_names == ["T_wall"]


def test_metal_node_heating_rate():
    # 1 kg steel, 1000 W heat, no cooling: dT/dt = 1000/500 = 2 K/s
    mn = MetalNode("wall", mass=1.0, cp_metal=500.0, initial_T=300.0)
    states = {"T_wall": 300.0}
    inputs = {"hot_side.Q_dot": 1000.0, "cool_side.Q_dot": 0.0}
    outputs = mn.compute_outputs(0.0, states, inputs)
    derivs = mn.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["T_wall"] - 2.0) < 1e-10


def test_metal_node_cooling_drops_temperature():
    mn = MetalNode("wall", mass=1.0, cp_metal=500.0)
    states = {"T_wall": 600.0}
    inputs = {"hot_side.Q_dot": 0.0, "cool_side.Q_dot": 500.0}
    outputs = mn.compute_outputs(0.0, states, inputs)
    derivs = mn.get_state_derivatives(0.0, states, inputs, outputs)
    assert derivs["T_wall"] < 0.0


def test_metal_node_thermal_equilibrium():
    mn = MetalNode("wall", mass=2.0, cp_metal=500.0)
    states = {"T_wall": 400.0}
    inputs = {"hot_side.Q_dot": 1000.0, "cool_side.Q_dot": 1000.0}
    outputs = mn.compute_outputs(0.0, states, inputs)
    derivs = mn.get_state_derivatives(0.0, states, inputs, outputs)
    assert abs(derivs["T_wall"]) < 1e-10


def test_metal_node_initialize():
    mn = MetalNode("wall", mass=1.0, initial_T=800.0)
    assert abs(mn._state_values["T_wall"] - 800.0) < 1e-10


# ── Pump tests ────────────────────────────────────────────────────────────────

from atha.components.pump import Pump, PumpMap


def _make_pump(name, dP=1e6, mdot=10.0, rpm=1000.0, eta=0.85):
    pm = PumpMap(mdot_design=mdot, dP_design=dP, omega_design=rpm * math.pi / 30.0,
                 eta_design=eta)
    fluid = IdealGasBackend(gamma=1.4, R=287.0)
    return Pump(name, diameter=0.1, pump_map=pm, fluid=fluid)


def _pump_inputs(rpm=1000.0, T=300.0, P=1e5, mdot=10.0):
    fluid = IdealGasBackend(gamma=1.4, R=287.0)
    h = fluid.cp * T
    return {"shaft.omega": rpm * math.pi / 30.0, "inlet.P": P, "inlet.h": h, "inlet.mdot": mdot}


def test_pump_no_states():
    p = _make_pump("p1")
    assert p.n_states == 0


def test_pump_design_point():
    p = _make_pump("p1", eta=1.0)
    inputs = _pump_inputs()
    out = p.compute_outputs(0.0, {}, inputs)
    assert abs(out["delta_P"] - 1e6) < 1.0  # design-point pressure rise


def test_pump_speed_scaling():
    p = _make_pump("p1")
    out_d = p.compute_outputs(0.0, {}, _pump_inputs(rpm=1000.0))
    out_h = p.compute_outputs(0.0, {}, _pump_inputs(rpm=500.0))
    # dP ∝ ω²: half speed → quarter pressure
    assert abs(out_h["delta_P"] - out_d["delta_P"] * 0.25) / out_d["delta_P"] < 1e-6


def test_pump_tau_positive():
    p = _make_pump("p1")
    out = p.compute_outputs(0.0, {}, _pump_inputs())
    assert out["tau_load"] > 0.0  # pump is a load on the shaft


def test_pump_solves_mdot_from_outlet_pressure():
    p = _make_pump("p1", dP=1e6, mdot=10.0, rpm=1000.0)
    inputs = _pump_inputs(rpm=1000.0)
    inputs.pop("inlet.mdot")
    inputs["outlet.P"] = inputs["inlet.P"] + 0.5e6  # half of shutoff head at design speed
    out = p.compute_outputs(0.0, {}, inputs)
    # With the built-in linear head model, design point sits near 50% runout => mdot ~ mdot_design
    assert abs(out["inlet.mdot"] - 10.0) < 1e-6
    assert abs(out["mdot"] - out["inlet.mdot"]) < 1e-12


def test_pump_solve_mdot_helper_matches_compute_outputs():
    p = _make_pump("p1", dP=1e6, mdot=10.0, rpm=1000.0)
    inputs = _pump_inputs(rpm=1000.0)
    P_in = inputs["inlet.P"]
    P_out = P_in + 0.6e6
    omega = inputs["shaft.omega"]
    mdot_helper = p.solve_mdot(P_in=P_in, P_out=P_out, omega=omega)
    out = p.compute_outputs(0.0, {}, {
        "shaft.omega": omega, "inlet.P": P_in, "inlet.h": inputs["inlet.h"], "outlet.P": P_out
    })
    assert abs(mdot_helper - out["inlet.mdot"]) < 1e-12


# ── Turbine tests ─────────────────────────────────────────────────────────────

from atha.components.turbine import Turbine, TurbineMap


def _make_turbine(name, eta=0.85, gamma=1.4):
    tm = TurbineMap(PR_design=2.0, eta_design=eta, mdot_corrected_design=1.0)
    return Turbine(name, diameter=0.1, turbine_map=tm, gamma=gamma)


def test_turbine_no_states():
    t = _make_turbine("t1")
    assert t.n_states == 0


def test_turbine_power_positive():
    t = _make_turbine("t1", eta=1.0)
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    P_in, P_out = 2e5, 1e5
    state_in = gas.state_from_PT(P_in, 1000.0)
    inputs = {"inlet.P": P_in, "inlet.h": state_in.h, "outlet.P": P_out,
              "inlet.mdot": 1.0, "shaft.omega": 100.0}
    out = t.compute_outputs(0.0, {}, inputs)
    assert out["power"] > 0.0
    assert out["tau_drive"] > 0.0


def test_turbine_isentropic_power():
    """For ideal gas, the ideal-gas approximation matches the thermo-based result exactly."""
    t = _make_turbine("t1", eta=1.0)
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    T_in, P_in, P_out = 1000.0, 2e5, 1e5
    state_in = gas.state_from_PT(P_in, T_in)
    # Expected: Δh_s = h_in * (1 - PR^(-(γ-1)/γ))
    PR = P_in / P_out
    exp = (1.4 - 1) / 1.4
    expected_dh = state_in.h * (1.0 - PR ** (-exp))
    inputs = {"inlet.P": P_in, "inlet.h": state_in.h, "outlet.P": P_out,
              "inlet.mdot": 2.0, "shaft.omega": 100.0}
    out = t.compute_outputs(0.0, {}, inputs)
    assert abs(out["delta_h"] - expected_dh) / expected_dh < 1e-6
    assert abs(out["power"] - 2.0 * expected_dh) / out["power"] < 1e-6


def test_turbine_efficiency_reduces_power():
    t_ideal = _make_turbine("t1", eta=1.0)
    t_real  = _make_turbine("t2", eta=0.85)
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    state_in = gas.state_from_PT(5e5, 1200.0)
    inputs = {"inlet.P": 5e5, "inlet.h": state_in.h, "outlet.P": 1e5,
              "inlet.mdot": 5.0, "shaft.omega": 200.0}
    out_i = t_ideal.compute_outputs(0.0, {}, inputs)
    out_r = t_real.compute_outputs(0.0, {}, inputs)
    assert abs(out_r["power"] - 0.85 * out_i["power"]) / out_i["power"] < 1e-6


# ── Nozzle tests ──────────────────────────────────────────────────────────────

from atha.components.nozzle import Nozzle


def test_nozzle_no_states():
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    n = Nozzle("n1", throat_area=0.01, exit_area=0.77, thermo=gas)
    assert n.n_states == 0


def test_nozzle_thrust_positive():
    gas = IdealGasBackend(gamma=1.2, R=520.0)
    n = Nozzle("n1", throat_area=0.0687, exit_area=5.29, thermo=gas,
               ambient_pressure=0.0, discharge_coeff=1.0,
               eta_velocity=1.0, eta_divergence=1.0)
    state_in = gas.state_from_PT(20e6, 3600.0)
    inputs = {"inlet.P": 20e6, "inlet.h": state_in.h, "inlet.mdot": 668.0}
    out = n.compute_outputs(0.0, {}, inputs)
    assert out["thrust"] > 0.0
    assert out["Cf"] > 1.0  # vacuum Cf for high expansion ratio


def test_nozzle_higher_Pc_higher_thrust():
    gas = IdealGasBackend(gamma=1.2, R=520.0)
    n = Nozzle("n1", throat_area=0.0687, exit_area=5.29, thermo=gas,
               ambient_pressure=0.0)
    state_lo = gas.state_from_PT(10e6, 3600.0)
    state_hi = gas.state_from_PT(20e6, 3600.0)
    out_lo = n.compute_outputs(0.0, {}, {"inlet.P": 10e6, "inlet.h": state_lo.h, "inlet.mdot": 334.0})
    out_hi = n.compute_outputs(0.0, {}, {"inlet.P": 20e6, "inlet.h": state_hi.h, "inlet.mdot": 668.0})
    assert out_hi["thrust"] > out_lo["thrust"]


# ── CombustionChamber tests ───────────────────────────────────────────────────

from atha.components.combustion_chamber import CombustionChamber


def test_combustion_chamber_has_two_states():
    gas = IdealGasBackend(gamma=1.2, R=520.0)
    cc = CombustionChamber("cc", volume=0.05, thermo=gas, T_adiabatic=3600.0,
                            initial_P=20e6, initial_T=3400.0)
    assert cc.n_states == 2
    assert "P" in cc.state_names
    assert "h" in cc.state_names


def test_combustion_chamber_pressure_rises_with_inflow():
    gas = IdealGasBackend(gamma=1.2, R=520.0)
    cc = CombustionChamber("cc", volume=0.05, thermo=gas, T_adiabatic=3600.0,
                            initial_P=20e6, initial_T=3400.0)
    state = gas.state_from_PT(20e6, 3400.0)
    states = {"P": 20e6, "h": state.h}
    inputs = {"fuel_inlet.mdot": 80.0, "ox_inlet.mdot": 480.0, "outlet.mdot": 550.0}
    outputs = cc.compute_outputs(0.0, states, inputs)
    derivs = cc.get_state_derivatives(0.0, states, inputs, outputs)
    # Net inflow > outflow → pressure rises
    assert derivs["P"] > 0.0


def test_combustion_chamber_initialize():
    gas = IdealGasBackend(gamma=1.2, R=520.0)
    cc = CombustionChamber("cc", volume=0.05, thermo=gas, T_adiabatic=3600.0)
    cc.initialize({"P": 15e6, "T": 3200.0})
    assert abs(cc._state_values["P"] - 15e6) < 1.0


# ── Preburner tests ───────────────────────────────────────────────────────────

from atha.components.preburner import Preburner


def test_preburner_is_combustion_chamber():
    gas = IdealGasBackend(gamma=1.3, R=350.0)
    pb = Preburner("pb", volume=0.01, thermo=gas, T_adiabatic=900.0,
                   initial_P=30e6, initial_T=800.0)
    assert pb.n_states == 2
    state = gas.state_from_PT(30e6, 800.0)
    states = {"P": 30e6, "h": state.h}
    inputs = {"fuel_inlet.mdot": 200.0, "ox_inlet.mdot": 20.0, "outlet.mdot": 210.0}
    outputs = pb.compute_outputs(0.0, states, inputs)
    derivs = pb.get_state_derivatives(0.0, states, inputs, outputs)
    assert isinstance(derivs["P"], float)
