from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np


@dataclass(frozen=True)
class ResidualDiagnostic:
    name: str
    value: float
    scale: float
    normalized_value: float


@dataclass(frozen=True)
class NonlinearDiagnostics:
    success: bool
    iterations: int
    norm: float
    residual_names: List[str]
    largest_residuals: List[ResidualDiagnostic]
    message: str


@dataclass(frozen=True)
class NonlinearSolveResult:
    z: np.ndarray
    residual: np.ndarray
    success: bool
    diagnostics: NonlinearDiagnostics


class NonlinearSolveError(RuntimeError):
    def __init__(self, diagnostics: NonlinearDiagnostics):
        largest = diagnostics.largest_residuals[0] if diagnostics.largest_residuals else None
        detail = ""
        if largest is not None:
            detail = (
                f"; largest residual {largest.name}="
                f"{largest.value:.6g} (normalized {largest.normalized_value:.6g})"
            )
        super().__init__(f"Nonlinear solve failed: {diagnostics.message}{detail}")
        self.diagnostics = diagnostics


def _diagnostics(
    residual: np.ndarray,
    scales: np.ndarray,
    names: List[str],
    success: bool,
    iterations: int,
    message: str,
) -> NonlinearDiagnostics:
    normalized = residual / scales
    order = np.argsort(-np.abs(normalized))
    largest = [
        ResidualDiagnostic(
            name=names[int(i)],
            value=float(residual[int(i)]),
            scale=float(scales[int(i)]),
            normalized_value=float(normalized[int(i)]),
        )
        for i in order[: min(5, len(order))]
    ]
    return NonlinearDiagnostics(
        success=success,
        iterations=iterations,
        norm=float(np.linalg.norm(normalized, ord=2)),
        residual_names=names,
        largest_residuals=largest,
        message=message,
    )


def _finite_difference_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    z: np.ndarray,
    residual: np.ndarray,
    variable_scales: np.ndarray,
) -> np.ndarray:
    jac = np.zeros((len(residual), len(z)))
    for j in range(len(z)):
        step = 1e-6 * max(abs(z[j]), variable_scales[j], 1.0)
        zp = z.copy()
        zp[j] += step
        jac[:, j] = (residual_fn(zp) - residual) / step
    return jac


def solve_nonlinear(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    residual_scales: Optional[np.ndarray] = None,
    variable_scales: Optional[np.ndarray] = None,
    residual_names: Optional[List[str]] = None,
    tol: float = 1e-8,
    max_iter: int = 25,
) -> NonlinearSolveResult:
    """
    Solve R(z)=0 with finite-difference Newton and simple damping.

    This is the initial Phase 1 nonlinear core: dense finite differences,
    residual scaling, variable scaling for perturbation sizes, and named
    diagnostics. Sparse patterns, bounds, and trust-region fallback can layer
    on this API without changing callers.
    """
    z = np.asarray(z0, dtype=float).copy()
    residual = np.asarray(residual_fn(z), dtype=float)
    scales = np.ones_like(residual) if residual_scales is None else np.asarray(residual_scales, dtype=float)
    var_scales = np.ones_like(z) if variable_scales is None else np.asarray(variable_scales, dtype=float)
    names = residual_names or [f"R[{i}]" for i in range(len(residual))]

    if np.any(scales <= 0.0):
        raise ValueError("residual_scales must be positive")
    if np.any(var_scales <= 0.0):
        raise ValueError("variable_scales must be positive")
    if len(names) != len(residual):
        raise ValueError("residual_names length must match residual length")

    for iteration in range(max_iter + 1):
        norm = np.linalg.norm(residual / scales, ord=2)
        if norm < tol:
            diagnostics = _diagnostics(residual, scales, names, True, iteration, "converged")
            return NonlinearSolveResult(z=z, residual=residual, success=True, diagnostics=diagnostics)
        if iteration == max_iter:
            break

        jac = _finite_difference_jacobian(residual_fn, z, residual, var_scales)
        jac_scaled = jac / scales[:, None]
        rhs_scaled = -(residual / scales)
        try:
            step = np.linalg.lstsq(jac_scaled, rhs_scaled, rcond=None)[0]
        except np.linalg.LinAlgError:
            break

        accepted = False
        alpha = 1.0
        while alpha >= 1e-4:
            trial_z = z + alpha * step
            trial_residual = np.asarray(residual_fn(trial_z), dtype=float)
            trial_norm = np.linalg.norm(trial_residual / scales, ord=2)
            if np.isfinite(trial_norm) and trial_norm < norm:
                z = trial_z
                residual = trial_residual
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    diagnostics = _diagnostics(residual, scales, names, False, max_iter, "maximum iterations or damping failure")
    raise NonlinearSolveError(diagnostics)
