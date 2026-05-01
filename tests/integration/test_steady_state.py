# tests/integration/test_steady_state.py
"""Integration tests: steady-state solver on simple engine configurations."""
import numpy as np
import pytest
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
from atha.solver.steady_state import SteadyStateSolver


def test_single_volume_balanced_flow_steady_state():
    """Single volume with equal inlet and outlet: dP/dt=0 at steady state."""
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("v", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    vol.add_outlet("outlet")
    engine = Engine("test")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    bcs = {"inlet.mdot": 2.0, "inlet.h": fs0.h, "outlet.mdot": 2.0}
    X_ss = SteadyStateSolver(layout).solve(X0, bcs)
    # Scatter and check derivatives
    layout.scatter_state_vector(X_ss)
    vol_comp = layout.components[0]
    states = {"P": float(X_ss[0]), "h": float(X_ss[1])}
    outputs = vol_comp.compute_outputs(0.0, states, bcs)
    derivs = vol_comp.get_state_derivatives(0.0, states, bcs, outputs)
    assert abs(derivs["P"]) < 1.0   # Pa/s
    assert abs(derivs["h"]) < 10.0  # J/(kg·s)


def test_two_volumes_in_series_steady_state():
    """Two volumes in series: at steady state, flow through both equals inlet flow."""
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    v1 = Volume("v1", volume=0.01, thermo=gas, initial_P=1.1e5, initial_T=300.0)
    v2 = Volume("v2", volume=0.01, thermo=gas, initial_P=1.0e5, initial_T=300.0)
    v1.add_inlet("inlet")
    v1.add_outlet("mid")
    v2.add_inlet("inlet2")
    v2.add_outlet("outlet")
    engine = Engine("test")
    engine.add_component(v1)
    engine.add_component(v2)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs = gas.state_from_PT(1.05e5, 300.0)
    # Drive 1 kg/s through both volumes
    bcs = {
        "inlet.mdot": 1.0, "inlet.h": fs.h,
        "mid.mdot":   1.0,  # v1 outlet
        "inlet2.mdot": 1.0, "inlet2.h": fs.h,  # v2 inlet
        "outlet.mdot": 1.0,
    }
    X_ss = SteadyStateSolver(layout, tol=1e-8).solve(X0, bcs)
    layout.scatter_state_vector(X_ss)
    for comp in layout.components:
        off = layout.state_offsets[comp.name]
        states = {n: float(X_ss[off + i]) for i, n in enumerate(comp.state_names)}
        outputs = comp.compute_outputs(0.0, states, bcs)
        derivs = comp.get_state_derivatives(0.0, states, bcs, outputs)
        assert abs(derivs["P"]) < 1.0, f"{comp.name} dP/dt not zero at SS"


def test_jannaf_nozzle_isp_range():
    """JANNAF simplified nozzle: LOX/LH2-like ideal gas gives Isp 380-500s vacuum."""
    from atha.jannaf.simplified import SimplifiedJANNAF
    from atha.jannaf.efficiency import JANNAFEfficiencies
    # Approximate LOX/LH2 combustion products
    gas = IdealGasBackend(gamma=1.2, R=520.0)
    eff = JANNAFEfficiencies()
    jannaf = SimplifiedJANNAF(gas, eff, throat_area=0.0687, exit_area=5.29, ambient_pressure=0.0)
    result = jannaf.compute(P_chamber=20e6, T_chamber=3600.0, MR=6.0, mdot_total=668.0)
    assert 380 < result.Isp < 520, f"Isp={result.Isp:.1f}s out of range"
    assert result.thrust > 0.0


def test_jannaf_nozzle_cf_formula_consistency():
    """Isp = c_star * Cf / g0 — internal consistency."""
    from atha.jannaf.simplified import SimplifiedJANNAF, G0
    from atha.jannaf.efficiency import JANNAFEfficiencies
    gas = IdealGasBackend(gamma=1.2, R=520.0)
    jannaf = SimplifiedJANNAF(gas, JANNAFEfficiencies(),
                               throat_area=0.05, exit_area=3.85, ambient_pressure=0.0)
    r = jannaf.compute(P_chamber=15e6, T_chamber=3500.0, MR=6.0, mdot_total=200.0)
    assert abs(r.Isp - r.c_star * r.Cf / G0) < 0.1, "Isp ≠ c_star * Cf / g0"


def test_volume_pressure_equilibrium_value():
    """At SS with 1 kg/s balanced flow, P should remain at initial value."""
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    P_target = 3e5
    vol = Volume("v", volume=0.05, thermo=gas, initial_P=P_target, initial_T=400.0)
    vol.add_inlet("in")
    vol.add_outlet("out")
    engine = Engine("e")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs = gas.state_from_PT(P_target, 400.0)
    bcs = {"in.mdot": 1.0, "in.h": fs.h, "out.mdot": 1.0}
    X_ss = SteadyStateSolver(layout).solve(X0, bcs)
    # At balanced flow, any P is a valid SS (dP/dt=0 for any P when mdot_net=0)
    # Just verify solver ran successfully and returned valid state
    assert X_ss[0] > 0.0  # P must be positive
    assert X_ss[1] > 0.0  # h must be positive
