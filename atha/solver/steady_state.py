from __future__ import annotations
import numpy as np
from scipy.optimize import root
from typing import Dict, Set

# Bare output keys that carry meaning across connections.
# Fluid/thermal scalars propagate by their own name; shaft keys have special aliases.
_BARE_FLUID = {"P", "h", "T", "rho", "gamma"}
_BARE_FLOW  = {"mdot"}
_BARE_SHAFT = {"omega"}
_BARE_TAU   = {"tau_drive", "tau_load"}
_PROPAGATABLE = _BARE_FLUID | _BARE_FLOW | _BARE_SHAFT | _BARE_TAU


def newton_solve(F, x0, tol=1e-10, max_iter=200):
    result = root(F, x0, method="hybr", tol=tol,
                  options={"maxfev": max_iter * (len(x0) + 1)})
    if not result.success:
        raise RuntimeError(f"Newton solver failed: {result.message}")
    return result.x


def _component_inputs(comp_name: str, context: Dict) -> Dict:
    """
    Build component-specific inputs from the global context dict.

    Keys prefixed with ``comp_name.`` are stripped to their suffix and added
    alongside the originals.  Non-prefixed keys pass through unchanged.

    Example:
        context key "lox_pump.inlet.P" → added as "inlet.P" for lox_pump.
        context key "gas.T"            → kept as "gas.T" for every component.
    """
    prefix = comp_name + "."
    inputs: Dict = {}
    for k, v in context.items():
        inputs[k] = v
        if k.startswith(prefix):
            inputs[k[len(prefix):]] = v
    return inputs


def _propagate_forward(conn, outputs: Dict, context: Dict, bcs: Dict) -> None:
    """
    Propagate src_comp outputs → dst_comp port inputs.

    Rules
    -----
    1. Port-prefixed keys  ``"{src_port}.X"``  → ``"{dst_comp}.{dst_port}.X"``
    2. Bare tau aliases    ``"tau_drive"`` / ``"tau_load"``  → ``"{dst_comp}.{dst_port}.tau"``
    3. Bare scalar keys    omega / mdot / P / h / T / rho / gamma
                           → ``"{dst_comp}.{dst_port}.{key}"``
    """
    src_prefix = conn.src_port + "."
    for k, v in outputs.items():
        if not isinstance(v, (int, float)):
            continue
        if k.startswith(src_prefix):
            dest_key = f"{conn.dst_comp}.{conn.dst_port}.{k[len(src_prefix):]}"
        elif k in _BARE_TAU:
            dest_key = f"{conn.dst_comp}.{conn.dst_port}.tau"
        elif k in (_BARE_FLUID | _BARE_FLOW | _BARE_SHAFT):
            dest_key = f"{conn.dst_comp}.{conn.dst_port}.{k}"
        else:
            continue
        if dest_key not in bcs:
            context[dest_key] = v


def _propagate_reverse(conn, outputs: Dict, context: Dict, bcs: Dict) -> None:
    """
    Propagate dst_comp outputs → src_comp port inputs (reverse direction).

    Same rules but roles are swapped: dst_port prefix stripped, src_port applied.
    Used so nozzle.mdot → chamber.outlet.mdot and shaft.omega → pump.shaft.omega.
    """
    dst_prefix = conn.dst_port + "."
    for k, v in outputs.items():
        if not isinstance(v, (int, float)):
            continue
        if k.startswith(dst_prefix):
            dest_key = f"{conn.src_comp}.{conn.src_port}.{k[len(dst_prefix):]}"
        elif k in _BARE_TAU:
            dest_key = f"{conn.src_comp}.{conn.src_port}.tau"
        elif k in (_BARE_FLUID | _BARE_FLOW | _BARE_SHAFT):
            dest_key = f"{conn.src_comp}.{conn.src_port}.{k}"
        else:
            continue
        if dest_key not in bcs:
            context[dest_key] = v


def _seed_from_states(layout, context: Dict, bcs: Dict) -> None:
    """
    Seed context with component state values propagated through connections.

    For each connection (src, src_port) → (dst, dst_port):
    - src dynamic states → ``"{dst_comp}.{dst_port}.{state}"``  (forward seeding)
    - dst dynamic states → ``"{src_comp}.{src_port}.{state}"``  (reverse seeding)

    This gives the Newton solver a physically consistent warm-start so that, e.g.,
    chamber.P is visible as ``lox_inj.outlet.P`` before any outputs are computed.
    """
    comp_map = {c.name: c for c in layout.components}
    for conn in layout.connections:
        src = comp_map[conn.src_comp]
        dst = comp_map[conn.dst_comp]
        for sname in src.state_names:
            val = src._state_values.get(sname)
            if val is not None:
                key = f"{conn.dst_comp}.{conn.dst_port}.{sname}"
                if key not in bcs and key not in context:
                    context[key] = val
        for sname in dst.state_names:
            val = dst._state_values.get(sname)
            if val is not None:
                key = f"{conn.src_comp}.{conn.src_port}.{sname}"
                if key not in bcs and key not in context:
                    context[key] = val


def _evaluate_pass(layout, X: np.ndarray, context: Dict, bcs: Dict,
                   collect_residuals: bool = False):
    """
    Single evaluation pass over all components.

    Propagates outputs into *context* (forward and reverse along connections).
    If *collect_residuals* is True, returns the residual vector; otherwise [].
    """
    resid = []
    for comp in layout.components:
        off = layout.state_offsets.get(comp.name)
        states: Dict = {}
        if off is not None:
            for i, sname in enumerate(comp.state_names):
                states[sname] = float(X[off + i])

        inputs = _component_inputs(comp.name, context)
        outputs = comp.compute_outputs(0.0, states, inputs)
        comp.last_outputs = outputs

        # Propagate outputs via connections
        for conn in layout.connections:
            if conn.src_comp == comp.name:
                _propagate_forward(conn, outputs, context, bcs)
            if conn.dst_comp == comp.name:
                _propagate_reverse(conn, outputs, context, bcs)

        if collect_residuals:
            derivs = comp.get_state_derivatives(0.0, states, inputs, outputs)
            for name in comp.state_names:
                resid.append(derivs.get(name, 0.0))
            alg_resid = comp.get_residuals(0.0, states, inputs, outputs)
            resid.extend(alg_resid.values())

    return resid


class SteadyStateSolver:
    def __init__(self, layout, tol=1e-8, max_iter=200):
        self.layout = layout
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, X0, boundary_conditions):
        """Find X* such that all dX/dt = 0 and all algebraic residuals = 0."""
        layout = self.layout
        bcs = dict(boundary_conditions)

        def residuals(X):
            layout.scatter_state_vector(X)

            # Build context: seed from states, then BCS overrides
            context: Dict = {}
            _seed_from_states(layout, context, bcs)
            context.update(bcs)

            # Pass 1 — propagate outputs; don't collect residuals yet
            _evaluate_pass(layout, X, context, bcs, collect_residuals=False)

            # Pass 2 — re-evaluate with fully populated context, collect residuals
            resid = _evaluate_pass(layout, X, context, bcs, collect_residuals=True)

            return np.array(resid) if resid else np.zeros(len(X))

        X_sol = newton_solve(residuals, X0, tol=self.tol, max_iter=self.max_iter)
        layout.scatter_state_vector(X_sol)
        # Final evaluation to populate last_outputs at the solution
        residuals(X_sol)
        return X_sol
