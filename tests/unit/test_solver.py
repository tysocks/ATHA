# tests/unit/test_solver.py
"""Unit tests for steady-state and transient solvers."""
import numpy as np
import pytest
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
from atha.solver.steady_state import newton_solve, SteadyStateSolver
from atha.solver.transient import TransientSolver, TransientSolution


# -- newton_solve --------------------------------------------------------------

def test_newton_solves_linear():
    def F(x):
        return np.array([x[0] - 3.0, x[1] + 2.0])
    sol = newton_solve(F, np.array([0.0, 0.0]))
    assert abs(sol[0] - 3.0) < 1e-8
    assert abs(sol[1] + 2.0) < 1e-8


def test_newton_solves_nonlinear():
    def F(x):
        return np.array([x[0]**2 + x[1]**2 - 4.0, x[0] - x[1]])
    sol = newton_solve(F, np.array([1.0, 1.0]))
    assert abs(sol[0] - np.sqrt(2.0)) < 1e-6
    assert abs(sol[1] - np.sqrt(2.0)) < 1e-6


def test_newton_raises_on_no_solution():
    def F(x):
        return np.array([x[0]**2 + 1.0])
    with pytest.raises(RuntimeError, match="Newton solver failed"):
        newton_solve(F, np.array([0.0]), max_iter=5)


# -- SteadyStateSolver --------------------------------------------------------

def _make_volume_engine(P0=1e5, T0=300.0, V=0.01):
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=V, thermo=gas, initial_P=P0, initial_T=T0)
    vol.add_inlet("inlet")
    vol.add_outlet("outlet")
    engine = Engine("test")
    engine.add_component(vol)
    return engine, gas


def test_steady_state_balanced_flow():
    engine, gas = _make_volume_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    bcs = {"inlet.mdot": 1.0, "inlet.h": fs0.h, "outlet.mdot": 1.0}
    solver = SteadyStateSolver(layout, tol=1e-10)
    X_ss = solver.solve(X0, bcs)
    layout.scatter_state_vector(X_ss)
    vol = layout.components[0]
    states = {"P": float(X_ss[0]), "h": float(X_ss[1])}
    outputs = vol.compute_outputs(0.0, states, bcs)
    derivs = vol.get_state_derivatives(0.0, states, bcs, outputs)
    assert abs(derivs["P"]) < 1.0


def test_steady_state_returns_array():
    engine, gas = _make_volume_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    bcs = {"inlet.mdot": 0.5, "inlet.h": fs0.h, "outlet.mdot": 0.5}
    solver = SteadyStateSolver(layout)
    X_ss = solver.solve(X0, bcs)
    assert isinstance(X_ss, np.ndarray)
    assert len(X_ss) == layout.n_states


# -- TransientSolver ----------------------------------------------------------

def test_transient_pressure_rises_with_inflow():
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    engine = Engine("test")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    def bcs(t):
        return {"inlet.mdot": 0.1, "inlet.h": fs0.h}
    solver = TransientSolver(layout, method="Radau", max_step=0.01)
    result = solver.integrate((0.0, 1.0), X0, bcs)
    P_series = result.get("vol", "P")
    assert P_series[-1] > P_series[0]


def test_transient_solution_get():
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    engine = Engine("test")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    def bcs(t):
        return {"inlet.mdot": 0.1, "inlet.h": fs0.h}
    solver = TransientSolver(layout, method="Radau", max_step=0.01)
    result = solver.integrate((0.0, 0.5), X0, bcs)
    P = result.get("vol", "P")
    assert isinstance(P, np.ndarray)
    assert len(P) == len(result.t)


def test_transient_solution_get_missing_raises():
    sol = TransientSolution(
        t=np.array([0.0, 1.0]),
        X=np.array([[1e5, 3e5], [1e5, 3e5]]),
        state_names=["vol.P", "vol.h"],
    )
    with pytest.raises(KeyError):
        sol.get("vol", "nonexistent")


def test_transient_balanced_flow_stable_pressure():
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    vol.add_outlet("outlet")
    engine = Engine("test")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    def bcs(t):
        return {"inlet.mdot": 1.0, "inlet.h": fs0.h, "outlet.mdot": 1.0}
    solver = TransientSolver(layout, method="Radau", max_step=0.01)
    result = solver.integrate((0.0, 1.0), X0, bcs)
    P_series = result.get("vol", "P")
    assert abs(P_series[-1] - P_series[0]) / P_series[0] < 0.01


def test_transient_state_names():
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas)
    vol.add_inlet("inlet")
    engine = Engine("test")
    engine.add_component(vol)
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(1e5, 300.0)
    def bcs(t):
        return {"inlet.mdot": 0.0, "inlet.h": fs0.h}
    solver = TransientSolver(layout, method="Radau", max_step=0.01)
    result = solver.integrate((0.0, 0.1), X0, bcs)
    assert result.state_names == layout.all_state_names()
    assert "vol.P" in result.state_names
    assert "vol.h" in result.state_names
