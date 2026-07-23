"""Legacy EngineLayout transient solver.

The canonical production path is ``atha.runner.dae_execution.DAEExecutionProblem``
with YAML phases, controllers, and residual/derivative contracts.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from atha.core.engine import EngineLayout
from atha.solver.steady_state import (
    _component_inputs,
    _seed_from_states,
    _propagate_forward,
    _propagate_reverse,
)
from atha.solver.nonlinear import solve_nonlinear


@dataclass
class TransientSolution:
    t: np.ndarray
    X: np.ndarray
    state_names: List[str]
    t_events: Optional[list] = None

    def get(self, component_name, state_name):
        key = f"{component_name}.{state_name}"
        for i, n in enumerate(self.state_names):
            if n == key:
                return self.X[:, i]
        raise KeyError(f"State '{key}' not found. Available: {self.state_names}")

    def without_initial_sample(self) -> "TransientSolution":
        """Return a copy without the solver's unresolved initial condition row."""

        if len(self.t) <= 1:
            return self
        return TransientSolution(
            t=self.t[1:].copy(),
            X=self.X[1:].copy(),
            state_names=list(self.state_names),
            t_events=self.t_events,
        )


class TransientSolver:
    def __init__(self, layout, method="Radau", rtol=1e-4, atol=1e-6, max_step=1e-3):
        self.layout = layout
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self._last_Z = np.zeros(layout.n_algebraic)

    def integrate(self, t_span, X0, boundary_conditions_fn, events=None):
        layout = self.layout

        def rhs(t, X):
            bcs = boundary_conditions_fn(t)
            Z = self._solve_algebraics(t, X, bcs)

            # Build context: seed from states, then BCS overrides
            context: Dict = {}
            _seed_from_states(layout, context, bcs)
            context.update(bcs)
            self._inject_algebraics(context, Z)

            # Pass 1 — propagate outputs
            for comp in layout.components:
                off = layout.state_offsets.get(comp.name)
                states: Dict = {}
                if off is not None:
                    for i, name in enumerate(comp.state_names):
                        states[name] = float(X[off + i])
                inputs = _component_inputs(comp.name, context)
                outputs = comp.compute_outputs(t, states, inputs)
                comp.last_outputs = outputs
                for conn in layout.connections:
                    if conn.src_comp == comp.name:
                        _propagate_forward(conn, outputs, context, bcs)
                    if conn.dst_comp == comp.name:
                        _propagate_reverse(conn, outputs, context, bcs)

            # Pass 2 — collect dXdt with full context
            dXdt = np.zeros_like(X)
            for comp in layout.components:
                off = layout.state_offsets.get(comp.name)
                states = {}
                if off is not None:
                    for i, name in enumerate(comp.state_names):
                        states[name] = float(X[off + i])
                inputs = _component_inputs(comp.name, context)
                outputs = comp.compute_outputs(t, states, inputs)
                comp.last_outputs = outputs
                derivs = comp.get_state_derivatives(t, states, inputs, outputs)
                if off is not None:
                    for i, name in enumerate(comp.state_names):
                        dXdt[off + i] = derivs.get(name, 0.0)
            return dXdt

        sol = solve_ivp(rhs, t_span, X0, method=self.method,
                        rtol=self.rtol, atol=self.atol, max_step=self.max_step,
                        dense_output=True, events=events)
        if not sol.success:
            raise RuntimeError(f"Transient integration failed: {sol.message}")

        t_events = [arr.tolist() for arr in sol.t_events] if sol.t_events is not None else None
        return TransientSolution(t=sol.t, X=sol.y.T, state_names=layout.all_state_names(),
                                 t_events=t_events)

    def _inject_algebraics(self, context, Z):
        for comp in self.layout.components:
            off = self.layout.alg_offsets.get(comp.name)
            if off is None:
                continue
            for i, name in enumerate(comp.algebraic_names):
                context[f"{comp.name}.{name}"] = float(Z[off + i])

    def _solve_algebraics(self, t, X, bcs):
        if self.layout.n_algebraic == 0:
            return np.zeros(0)

        probe = self.layout.evaluate(t, X, self._last_Z, bcs)
        if len(probe.Rz) != self.layout.n_algebraic:
            # Connection residuals currently expand Rz beyond component Z.
            # Leave full DAE connection solving to the square assembled system.
            return self._last_Z

        result = solve_nonlinear(
            residual_fn=lambda Z: self.layout.evaluate(t, X, Z, bcs).Rz,
            z0=self._last_Z,
            residual_scales=probe.residual_scales,
            variable_scales=np.ones(self.layout.n_algebraic),
            residual_names=probe.residual_names,
            tol=1e-10,
            max_iter=20,
        )
        self._last_Z = result.z
        return result.z
