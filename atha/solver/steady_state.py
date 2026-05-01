from __future__ import annotations
import numpy as np
from scipy.optimize import root
from typing import Callable, Dict
from atha.core.engine import EngineLayout


def newton_solve(F, x0, tol=1e-10, max_iter=200):
    result = root(F, x0, method="hybr", tol=tol,
                  options={"maxfev": max_iter * (len(x0) + 1)})
    if not result.success:
        raise RuntimeError(f"Newton solver failed: {result.message}")
    return result.x


class SteadyStateSolver:
    def __init__(self, layout, tol=1e-8, max_iter=200):
        self.layout = layout
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, X0, boundary_conditions):
        layout = self.layout

        def residuals(X):
            layout.scatter_state_vector(X)
            resid = []
            for comp in layout.components:
                off = layout.state_offsets.get(comp.name)
                states = {}
                if off is not None:
                    for i, name in enumerate(comp.state_names):
                        states[name] = float(X[off + i])
                inputs = dict(boundary_conditions)
                outputs = comp.compute_outputs(0.0, states, inputs)
                derivs = comp.get_state_derivatives(0.0, states, inputs, outputs)
                for name in comp.state_names:
                    resid.append(derivs.get(name, 0.0))
                alg_resid = comp.get_residuals(0.0, states, inputs, outputs)
                resid.extend(alg_resid.values())
            return np.array(resid) if resid else np.zeros(len(X))

        X_sol = newton_solve(residuals, X0, tol=self.tol, max_iter=self.max_iter)
        layout.scatter_state_vector(X_sol)
        return X_sol
