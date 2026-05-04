# atha/profiles/executor.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np

from atha.profiles.phase import PhaseDefinition, PhaseMode
from atha.profiles.limits import SafetyLimit, AbortManager, EngineAbort
from atha.profiles.recording import downsample_dense_output
from atha.profiles.result import PhaseResult
from atha.solver.steady_state import SteadyStateSolver
from atha.solver.transient import TransientSolver


def execute_phase(
    layout,
    X0: np.ndarray,
    phase: PhaseDefinition,
    global_limits: List[SafetyLimit],
    extra_bcs_fn: Optional[Callable[[float], Dict[str, float]]] = None,
) -> PhaseResult:
    """Execute one phase of a test profile and return recorded data.

    Raises EngineAbort if a hard safety limit is exceeded.
    """
    all_limits = global_limits + list(phase.abort_checks)
    abort_mgr = AbortManager(all_limits)
    state_names = layout.all_state_names()

    if phase.mode == PhaseMode.STEADY_TRIM:
        return _execute_steady_trim(layout, X0, phase, abort_mgr, state_names)
    elif phase.mode == PhaseMode.TRANSIENT:
        return _execute_transient(layout, X0, phase, abort_mgr, state_names, extra_bcs_fn)
    elif phase.mode == PhaseMode.DWELL:
        return _execute_dwell(X0, phase, state_names)
    else:
        raise ValueError(f"Unknown PhaseMode: {phase.mode}")


def _execute_steady_trim(layout, X0, phase, abort_mgr, state_names) -> PhaseResult:
    solver = SteadyStateSolver(
        layout,
        tol=phase.solver_options.get("tol", 1e-8),
        max_iter=phase.solver_options.get("max_iter", 200),
    )
    X_sol = solver.solve(X0, phase.trim_targets)
    abort_mgr.check(layout, X_sol, t=phase.duration)

    t = np.array([0.0, phase.duration])
    X = np.vstack([X0, X_sol])
    return PhaseResult(
        name=phase.name,
        t=t, X=X,
        state_names=state_names,
        X_final=X_sol.copy(),
        abort_triggered=False,
    )


def _execute_transient(
    layout,
    X0,
    phase,
    abort_mgr,
    state_names,
    extra_bcs_fn: Optional[Callable[[float], Dict[str, float]]] = None,
) -> PhaseResult:
    def bcs(t_phase):
        result: Dict[str, float] = {}
        for cmd in phase.control_commands:
            result[cmd.bcs_key] = cmd.fn(t_phase)
        if extra_bcs_fn is not None:
            result.update(extra_bcs_fn(t_phase))
        return result

    # Build scipy event callbacks for limit checking
    scipy_events = abort_mgr.as_scipy_events(layout)

    rtol = phase.solver_options.get("rtol", 1e-4)
    atol = phase.solver_options.get("atol", 1e-6)
    max_step = phase.solver_options.get("max_step", min(0.01, phase.duration / 100))

    solver = TransientSolver(layout, method="Radau",
                             rtol=rtol, atol=atol, max_step=max_step)

    try:
        sol = solver.integrate(
            t_span=(0.0, phase.duration),
            X0=X0,
            boundary_conditions_fn=bcs,
            events=scipy_events if scipy_events else None,
        )
    except RuntimeError as e:
        raise EngineAbort(reason=f"Integrator failed: {e}", t=0.0) from e

    # Check if a hard limit event fired (terminal=True means integration stopped early)
    if scipy_events and sol.t_events is not None:
        for i, t_ev_list in enumerate(sol.t_events):
            if len(t_ev_list) > 0 and scipy_events[i].terminal:
                abort_t = float(t_ev_list[0])
                raise EngineAbort(
                    reason=f"Hard limit triggered at t={abort_t:.4f}s",
                    t=abort_t,
                )

    phase_result = downsample_dense_output(
        sol, phase.duration, phase.recording_rate_hz, phase_name=phase.name
    )
    return phase_result


def _execute_dwell(X0, phase, state_names) -> PhaseResult:
    t = np.array([0.0, phase.duration])
    X = np.vstack([X0, X0])
    return PhaseResult(
        name=phase.name,
        t=t, X=X,
        state_names=state_names,
        X_final=X0.copy(),
        abort_triggered=False,
    )
