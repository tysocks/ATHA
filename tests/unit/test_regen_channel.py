# tests/unit/test_regen_channel.py
import math
import pytest
from atha.components.regen_channel import RegenChannel, _friction_factor
from atha.thermo.ideal_gas import IdealGasBackend


# ── friction factor ───────────────────────────────────────────────────────────

def test_friction_factor_laminar():
    # f = 64/Re for Re < 2300
    assert abs(_friction_factor(1000) - 64.0 / 1000) < 1e-10
    assert abs(_friction_factor(2000) - 64.0 / 2000) < 1e-10


def test_friction_factor_turbulent_smooth():
    # At Re=1e5, smooth pipe: f ≈ 0.018 (Moody chart reference)
    f = _friction_factor(1e5, rel_roughness=0.0)
    assert 0.015 < f < 0.020


def test_friction_factor_decreases_with_re():
    f1 = _friction_factor(1e4)
    f2 = _friction_factor(1e5)
    assert f1 > f2


# ── component construction ────────────────────────────────────────────────────

def make_regen(initial_T_wall=300.0):
    fluid = IdealGasBackend(gamma=1.3, R=518.0, mu=1e-5, k=0.05, MW=0.016)
    return RegenChannel(
        "regen",
        fluid=fluid,
        channel_area=5e-5,
        hydraulic_diam=3e-3,
        channel_length=0.8,
        hot_area=0.15,
        cool_area=0.18,
        wall_mass=2.5,
        wall_cp=500.0,
        h_hot_design=5e4,
        Pc_design=10e6,
        recovery_factor=0.90,
        initial_T_wall=initial_T_wall,
    )


def test_regen_has_one_state():
    regen = make_regen()
    assert regen.n_states == 1
    assert "T_wall" in regen.state_names


def test_regen_has_correct_ports():
    regen = make_regen()
    assert "coolant_inlet"  in regen.ports
    assert "coolant_outlet" in regen.ports


# ── compute_outputs ───────────────────────────────────────────────────────────

def base_inputs(T_wall=500.0):
    fluid = IdealGasBackend(gamma=1.3, R=518.0, mu=1e-5, k=0.05, MW=0.016)
    h_in = fluid.state_from_PT(10e6, 150.0).h
    return {
        "coolant_inlet.mdot": 1.0,
        "coolant_inlet.P":    10e6,
        "coolant_inlet.h":    h_in,
        "gas.T":              3500.0,
        "gas.P":              10e6,
    }, T_wall


def test_coolant_enthalpy_rises():
    regen = make_regen(initial_T_wall=600.0)
    inputs, T_wall = base_inputs(T_wall=600.0)
    out = regen.compute_outputs(0.0, {"T_wall": T_wall}, inputs)
    assert out["coolant_outlet.h"] > inputs["coolant_inlet.h"]


def test_pressure_drops():
    regen = make_regen(initial_T_wall=600.0)
    inputs, T_wall = base_inputs(T_wall=600.0)
    out = regen.compute_outputs(0.0, {"T_wall": T_wall}, inputs)
    assert out["coolant_outlet.P"] < inputs["coolant_inlet.P"]
    assert out["delta_P"] > 0.0


def test_q_cool_positive_when_wall_hot():
    regen = make_regen(initial_T_wall=800.0)
    inputs, T_wall = base_inputs(T_wall=800.0)
    out = regen.compute_outputs(0.0, {"T_wall": T_wall}, inputs)
    assert out["Q_cool"] > 0.0


def test_q_hot_positive_when_gas_hotter_than_wall():
    regen = make_regen(initial_T_wall=300.0)
    inputs, T_wall = base_inputs(T_wall=300.0)
    out = regen.compute_outputs(0.0, {"T_wall": T_wall}, inputs)
    # T_aw = 0.9 * 3500 = 3150 K >> T_wall = 300 K → Q_hot > 0
    assert out["Q_hot"] > 0.0


def test_q_cool_negative_when_wall_colder_than_coolant():
    # When T_wall < T_coolant the coolant heats the wall (Q_cool < 0).
    # The unclamped formula is intentional: it keeps the residual smooth for Newton.
    regen = make_regen(initial_T_wall=50.0)
    inputs, T_wall = base_inputs(T_wall=50.0)
    out = regen.compute_outputs(0.0, {"T_wall": T_wall}, inputs)
    assert out["Q_cool"] < 0.0


# ── state derivatives ─────────────────────────────────────────────────────────

def test_wall_heats_up_when_q_hot_exceeds_q_cool():
    regen = make_regen(initial_T_wall=300.0)
    inputs, T_wall = base_inputs(T_wall=300.0)
    out = regen.compute_outputs(0.0, {"T_wall": T_wall}, inputs)
    derivs = regen.get_state_derivatives(0.0, {"T_wall": T_wall}, inputs, out)
    # At low T_wall, Q_hot >> Q_cool → wall should heat up
    assert derivs["T_wall"] > 0.0


def test_wall_cools_when_q_cool_exceeds_q_hot():
    regen = make_regen(initial_T_wall=4000.0)   # wall hotter than T_aw
    inputs, T_wall = base_inputs(T_wall=4000.0)
    inputs["gas.T"] = 500.0   # cool gas
    out = regen.compute_outputs(0.0, {"T_wall": T_wall}, inputs)
    derivs = regen.get_state_derivatives(0.0, {"T_wall": T_wall}, inputs, out)
    assert derivs["T_wall"] < 0.0


def test_steady_state_wall_temperature():
    """At steady state dT_wall/dt = 0 → Q_hot = Q_cool."""
    regen = make_regen(initial_T_wall=1200.0)
    inputs, _ = base_inputs()

    # Binary search for T_wall where dT/dt ≈ 0
    lo, hi = 300.0, 3000.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        out = regen.compute_outputs(0.0, {"T_wall": mid}, inputs)
        d = regen.get_state_derivatives(0.0, {"T_wall": mid}, inputs, out)
        if d["T_wall"] > 0:
            lo = mid
        else:
            hi = mid

    T_ss = (lo + hi) / 2.0
    out_ss = regen.compute_outputs(0.0, {"T_wall": T_ss}, inputs)
    assert abs(out_ss["Q_hot"] - out_ss["Q_cool"]) / max(out_ss["Q_hot"], 1.0) < 1e-3


# ── Bartz pressure scaling ─────────────────────────────────────────────────────

def test_h_hot_scales_with_pressure():
    regen = make_regen(initial_T_wall=500.0)
    inputs_lo = dict(base_inputs(500.0)[0]); inputs_lo["gas.P"] = 5e6
    inputs_hi = dict(base_inputs(500.0)[0]); inputs_hi["gas.P"] = 20e6
    out_lo = regen.compute_outputs(0.0, {"T_wall": 500.0}, inputs_lo)
    out_hi = regen.compute_outputs(0.0, {"T_wall": 500.0}, inputs_hi)
    # Higher Pc → higher Bartz h_hot
    assert out_hi["h_hot_coeff"] > out_lo["h_hot_coeff"]
    # Check exponent: h_hot ∝ P^0.8
    ratio = out_hi["h_hot_coeff"] / out_lo["h_hot_coeff"]
    expected = (20e6 / 5e6) ** 0.8
    assert abs(ratio - expected) < 0.01


# ── Dittus-Boelter scaling ────────────────────────────────────────────────────

def test_h_cool_increases_with_flow():
    regen = make_regen(initial_T_wall=600.0)
    inputs_lo = dict(base_inputs(600.0)[0]); inputs_lo["coolant_inlet.mdot"] = 0.5
    inputs_hi = dict(base_inputs(600.0)[0]); inputs_hi["coolant_inlet.mdot"] = 2.0
    out_lo = regen.compute_outputs(0.0, {"T_wall": 600.0}, inputs_lo)
    out_hi = regen.compute_outputs(0.0, {"T_wall": 600.0}, inputs_hi)
    assert out_hi["h_cool_coeff"] > out_lo["h_cool_coeff"]
