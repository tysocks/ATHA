"""Legacy EngineLayout steady-state helpers.

The canonical production path is ``atha.runner.dae_execution.DAEExecutionProblem``
with ``PortNetworkBuilder``. Prefer that stack for new models and examples.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set
from atha.solver.nonlinear import solve_nonlinear, NonlinearSolveError


def newton_solve(F, x0, tol=1e-10, max_iter=200):
    """Backward-compatible wrapper around solve_nonlinear."""
    z0 = np.asarray(x0, dtype=float)
    try:
        result = solve_nonlinear(
            residual_fn=F,
            z0=z0,
            tol=tol,
            max_iter=max_iter,
        )
        return result.z
    except (NonlinearSolveError, Exception) as exc:
        raise RuntimeError(f"Newton solver failed: {exc}") from exc

# Bare output keys that carry meaning across connections.
# Fluid/thermal scalars propagate by their own name; shaft keys have special aliases.
_BARE_FLUID = {"P", "h", "T", "rho", "gamma"}
_BARE_FLOW  = {"mdot"}
_BARE_SHAFT = {"omega"}
_BARE_TAU   = {"tau_drive", "tau_load"}
_PROPAGATABLE = _BARE_FLUID | _BARE_FLOW | _BARE_SHAFT | _BARE_TAU


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
    If *collect_residuals* is True, returns (resid_values, resid_names, deriv_scales).
    deriv_scales are |dX/dt| magnitudes used as fallback scales for components that
    do not implement get_steady_state_residuals.
    """
    resid_vals: List[float] = []
    resid_names: List[str] = []
    deriv_scales: List[float] = []

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
            ss_resid = comp.get_steady_state_residuals(0.0, states, inputs, outputs)
            if ss_resid is not None:
                # Dimensionless residuals provided by the component
                for sname in comp.state_names:
                    r = ss_resid.get(sname, 0.0)
                    resid_vals.append(r)
                    resid_names.append(f"{comp.name}.{sname}")
                    deriv_scales.append(1.0)
            else:
                # Fallback: use raw dX/dt, scale by magnitude at first call
                derivs = comp.get_state_derivatives(0.0, states, inputs, outputs)
                for sname in comp.state_names:
                    d = derivs.get(sname, 0.0)
                    resid_vals.append(d)
                    resid_names.append(f"{comp.name}.{sname}")
                    deriv_scales.append(max(abs(d), 1.0))

            alg_resid = comp.get_residuals(0.0, states, inputs, outputs)
            for aname, aval in alg_resid.items():
                resid_vals.append(aval)
                resid_names.append(f"{comp.name}.{aname}")
                deriv_scales.append(max(abs(aval), 1.0))

    return resid_vals, resid_names, deriv_scales


class SteadyStateSolver:
    def __init__(self, layout, tol=1e-8, max_iter=200):
        self.layout = layout
        self.tol = tol
        self.max_iter = max_iter

    def _build_residual_fn(self, layout, bcs):
        """Return (residual_fn, residual_names, residual_scales)."""
        scales_ref: List[float] = []
        names_ref: List[str] = []
        initialized = [False]

        # Build a map from state index → state name for physical clamping
        state_names = layout.all_state_names()

        def _clamp_to_physical(X: np.ndarray) -> np.ndarray:
            """Clamp state vector to physically admissible bounds."""
            Xc = X.copy()
            for i, name in enumerate(state_names):
                if name.endswith(".P"):
                    Xc[i] = max(Xc[i], 1.0)       # pressure >= 1 Pa
                elif name.endswith(".h"):
                    # Cantera JANAF enthalpies are negative for combustion products;
                    # only reject values more negative than -1 GJ/kg (clearly unphysical).
                    Xc[i] = max(Xc[i], -1e9)
                elif name.endswith(".omega"):
                    Xc[i] = max(Xc[i], 0.01)      # shaft speed >= 0.01 rad/s
                elif name.endswith(".T_wall"):
                    Xc[i] = max(Xc[i], 50.0)      # wall temp >= 50 K
            return Xc

        def residuals(X):
            Xc = _clamp_to_physical(X)
            layout.scatter_state_vector(Xc)

            context: Dict = {}
            _seed_from_states(layout, context, bcs)
            context.update(bcs)

            # Pass 1 — propagate outputs
            _evaluate_pass(layout, Xc, context, bcs, collect_residuals=False)

            # Pass 2 — re-evaluate with full context, collect residuals
            vals, names, dscales = _evaluate_pass(layout, Xc, context, bcs,
                                                  collect_residuals=True)

            if not initialized[0]:
                names_ref.extend(names)
                # For dimensionless residuals (scale=1.0 from ss_resid path),
                # keep scale=1.0.  For raw dX/dt fallback, use initial magnitude.
                scales_ref.extend(dscales)
                initialized[0] = True

            if not vals:
                return np.zeros(len(X))
            return np.array(vals, dtype=float)

        return residuals, names_ref, scales_ref

    def solve(self, X0, boundary_conditions):
        """Find X* such that all dX/dt = 0 and all algebraic residuals = 0."""
        layout = self.layout
        bcs = dict(boundary_conditions)

        residual_fn, names_ref, scales_ref = self._build_residual_fn(layout, bcs)

        # Prime the initializer with a first call
        R0 = residual_fn(X0.copy())

        if len(R0) == 0:
            # No states — nothing to solve
            layout.scatter_state_vector(X0)
            residual_fn(X0)
            return X0

        scales = np.array(scales_ref, dtype=float)
        # Clamp scales to avoid division by zero
        scales = np.where(scales > 0, scales, 1.0)

        result = solve_nonlinear(
            residual_fn=residual_fn,
            z0=X0.copy(),
            residual_scales=scales,
            variable_scales=np.ones(len(X0)),
            residual_names=names_ref,
            tol=self.tol,
            max_iter=self.max_iter,
        )

        X_sol = result.z
        layout.scatter_state_vector(X_sol)
        # Final evaluation to populate last_outputs at the solution
        residual_fn(X_sol)
        return X_sol
