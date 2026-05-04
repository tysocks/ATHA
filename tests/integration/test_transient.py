# tests/integration/test_transient.py
"""Integration tests: transient solver on dynamic engine scenarios."""
import numpy as np
import pytest
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.components.rotor import Rotor
from atha.core.engine import Engine
from atha.solver.transient import TransientSolver


def test_pressure_rise_transient():
    """Volume with 0.1 kg/s net inflow: pressure must increase monotonically."""
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    engine = Engine("e")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    def bcs(t):
        return {"inlet.mdot": 0.1, "inlet.h": fs0.h}
    solver = TransientSolver(layout, method="Radau", max_step=0.05)
    result = solver.integrate((0.0, 2.0), X0, bcs)
    P = result.get("vol", "P")
    assert P[-1] > P[0], "Pressure did not rise with net inflow"
    assert all(np.diff(P) >= -1.0), "Pressure should be monotonically non-decreasing"


def test_pressure_decay_transient():
    """Volume with 0.1 kg/s net outflow: pressure must decrease."""
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas, initial_P=2e5, initial_T=300.0)
    vol.add_outlet("outlet")
    engine = Engine("e")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    def bcs(t):
        return {"outlet.mdot": 0.01}
    solver = TransientSolver(layout, method="Radau", max_step=0.05)
    result = solver.integrate((0.0, 1.0), X0, bcs)
    P = result.get("vol", "P")
    assert P[-1] < P[0], "Pressure did not fall with net outflow"


def test_rotor_spin_up_transient():
    """Rotor under constant drive torque: omega increases from zero."""
    rotor = Rotor("rotor", moment_of_inertia=5.0, friction_coeff=0.1)
    rotor.port("turbine_in")  # drive (INLET)
    rotor.port("pump_out")    # load  (OUTLET)
    engine = Engine("e")
    engine.add_component(rotor)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    def bcs(t):
        return {"turbine_in.tau": 50.0, "pump_out.tau": 0.0}
    solver = TransientSolver(layout, method="Radau", max_step=0.05)
    result = solver.integrate((0.0, 5.0), X0, bcs)
    omega = result.get("rotor", "omega")
    assert omega[-1] > omega[0], "Rotor did not spin up"
    # Steady-state omega = tau_drive / k_f = 50 / 0.1 = 500 rad/s
    assert omega[-1] < 600.0  # should approach but not exceed SS value


def test_rotor_reaches_steady_state():
    """Rotor settles to omega = tau_net / friction_coeff."""
    rotor = Rotor("rotor", moment_of_inertia=1.0, friction_coeff=2.0)
    rotor.port("turbine_in")
    rotor.port("pump_out")
    engine = Engine("e")
    engine.add_component(rotor)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    tau_drive = 100.0
    k_f = 2.0
    omega_ss = tau_drive / k_f  # 50 rad/s
    def bcs(t):
        return {"turbine_in.tau": tau_drive, "pump_out.tau": 0.0}
    solver = TransientSolver(layout, method="Radau", max_step=0.05)
    result = solver.integrate((0.0, 20.0), X0, bcs)
    omega = result.get("rotor", "omega")
    # After 20 s, should be within 5% of steady-state
    assert abs(omega[-1] - omega_ss) / omega_ss < 0.05


def test_two_component_transient():
    """Two volumes: inflow to v1, outflow from v2. Both respond consistently."""
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    v1 = Volume("v1", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    v2 = Volume("v2", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    v1.add_inlet("inlet")
    v2.add_outlet("outlet")
    engine = Engine("e")
    engine.add_component(v1)
    engine.add_component(v2)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    def bcs(t):
        # Small flow rate (0.001 kg/s) keeps volume far from draining in 0.2 s
        return {"inlet.mdot": 0.001, "inlet.h": fs0.h, "outlet.mdot": 0.001}
    solver = TransientSolver(layout, method="Radau", max_step=0.05)
    result = solver.integrate((0.0, 0.2), X0, bcs)
    P_v1 = result.get("v1", "P")
    P_v2 = result.get("v2", "P")
    assert P_v1[-1] > P_v1[0]   # v1 receives flow, P rises
    assert P_v2[-1] < P_v2[0]   # v2 loses flow, P falls
