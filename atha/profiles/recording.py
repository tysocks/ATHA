# atha/profiles/recording.py
from __future__ import annotations
import numpy as np
from atha.profiles.result import PhaseResult
from atha.solver.transient import TransientSolution


def downsample_dense_output(
    sol: TransientSolution,
    t_duration: float,
    recording_rate_hz: float,
    phase_name: str = "",
) -> PhaseResult:
    """Resample TransientSolution at a uniform rate using linear interpolation.

    Args:
        sol: Result from TransientSolver.integrate()
        t_duration: Total duration of the phase [s]
        recording_rate_hz: Target recording rate [Hz]
        phase_name: Name for the returned PhaseResult

    Returns:
        PhaseResult with uniform time grid
    """
    n_points = int(round(t_duration * recording_rate_hz)) + 1
    t_record = np.linspace(0.0, t_duration, n_points)

    # Clamp to actual solution range
    t_record = np.clip(t_record, sol.t[0], sol.t[-1])

    # Interpolate each state column
    n_states = sol.X.shape[1]
    X_record = np.zeros((len(t_record), n_states))
    for i in range(n_states):
        X_record[:, i] = np.interp(t_record, sol.t, sol.X[:, i])

    return PhaseResult(
        name=phase_name,
        t=t_record,
        X=X_record,
        state_names=sol.state_names,
        X_final=X_record[-1].copy(),
        abort_triggered=False,
    )
