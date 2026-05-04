from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from atha.core.engine import EngineLayout


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


class TransientSolver:
    def __init__(self, layout, method="Radau", rtol=1e-4, atol=1e-6, max_step=1e-3):
        self.layout = layout
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

    def integrate(self, t_span, X0, boundary_conditions_fn, events=None):
        layout = self.layout

        def rhs(t, X):
            layout.scatter_state_vector(X)
            bcs = boundary_conditions_fn(t)
            dXdt = np.zeros_like(X)
            for comp in layout.components:
                off = layout.state_offsets.get(comp.name)
                states = {}
                if off is not None:
                    for i, name in enumerate(comp.state_names):
                        states[name] = float(X[off + i])
                inputs = dict(bcs)
                outputs = comp.compute_outputs(t, states, inputs)
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
