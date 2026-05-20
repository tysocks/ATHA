from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, root


@dataclass(frozen=True)
class NetworkVariable:
    """Named algebraic unknown in the global network vector Z."""

    name: str
    units: str = ""
    scale: float = 1.0
    initial: float = 0.0
    owner: str = ""
    description: str = ""
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class NetworkResidual:
    """Named residual equation in the global algebraic residual vector Rz."""

    name: str
    units: str = ""
    scale: float = 1.0
    owner: str = ""
    description: str = ""


@dataclass(frozen=True)
class ResidualDiagnostic:
    name: str
    value: float
    normalized: float
    scale: float


@dataclass
class NetworkSolution:
    z: np.ndarray
    values: dict[str, float]
    residuals: dict[str, float]
    normalized_residuals: dict[str, float]
    success: bool
    message: str

    @property
    def max_normalized_residual(self) -> tuple[str, float]:
        if not self.normalized_residuals:
            return "", 0.0
        name = max(self.normalized_residuals, key=lambda key: abs(self.normalized_residuals[key]))
        return name, float(self.normalized_residuals[name])

    def diagnostics(self) -> list[ResidualDiagnostic]:
        diagnostics = [
            ResidualDiagnostic(
                name=name,
                value=float(self.residuals[name]),
                normalized=float(normalized),
                scale=abs(float(self.residuals[name]) / float(normalized)) if normalized else 1.0,
            )
            for name, normalized in self.normalized_residuals.items()
        ]
        return sorted(diagnostics, key=lambda item: abs(item.normalized), reverse=True)


class NetworkSolveError(RuntimeError):
    """Raised when a checked network solve fails residual tolerance."""

    def __init__(self, problem_name: str, solution: NetworkSolution, tolerance: float) -> None:
        largest = solution.max_normalized_residual
        super().__init__(
            f"network solve '{problem_name}' failed residual tolerance {tolerance:g}; "
            f"success={solution.success}; largest residual {largest[0]}={largest[1]:.3e}; "
            f"message={solution.message}"
        )
        self.problem_name = problem_name
        self.solution = solution
        self.tolerance = tolerance


@dataclass
class WarmStart:
    """Mutable warm-start store for repeated algebraic solves during transients."""

    z: np.ndarray

    @classmethod
    def from_problem(cls, problem: "NetworkProblem") -> "WarmStart":
        return cls(problem.initial_z.copy())

    def update(self, solution: NetworkSolution) -> None:
        self.z = np.asarray(solution.z, dtype=float).copy()


class NetworkStructureError(ValueError):
    """Raised when the configured algebraic network is structurally invalid."""


ResidualEvaluator = Callable[[float, Mapping[str, float], Mapping[str, Any]], Mapping[str, float]]


class NetworkProblem:
    """Named, scaled algebraic network problem.

    This is the Phase-10 foundation for ATHA's global DAE solve. It owns the
    algebraic vector shape, residual vector shape, scaling, warm-start solve
    behavior, and failure diagnostics. Component-level residual contracts will
    feed this object in later phases.
    """

    def __init__(
        self,
        variables: Sequence[NetworkVariable],
        residuals: Sequence[NetworkResidual],
        evaluator: ResidualEvaluator,
        *,
        name: str = "network",
        sparse_jacobian_hint: Any | None = None,
        strict_residuals: bool = True,
        require_square: bool = True,
    ) -> None:
        self.name = name
        self.variables = list(variables)
        self.residual_definitions = list(residuals)
        self._evaluator = evaluator
        self.sparse_jacobian_hint = sparse_jacobian_hint
        self.strict_residuals = strict_residuals
        self.require_square = require_square
        self.variable_names = [variable.name for variable in self.variables]
        self.residual_names = [residual.name for residual in self.residual_definitions]
        self._validate_structure()
        self.variable_scales = np.array([max(abs(variable.scale), 1.0e-30) for variable in self.variables], dtype=float)
        self.residual_scales = np.array([max(abs(residual.scale), 1.0e-30) for residual in self.residual_definitions], dtype=float)
        self.initial_z = np.array([float(variable.initial) for variable in self.variables], dtype=float)
        self.lower_bounds = np.array(
            [-np.inf if variable.lower is None else float(variable.lower) for variable in self.variables],
            dtype=float,
        )
        self.upper_bounds = np.array(
            [np.inf if variable.upper is None else float(variable.upper) for variable in self.variables],
            dtype=float,
        )

    @property
    def n_algebraic(self) -> int:
        return len(self.variables)

    def values_from_z(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=float)
        if z.shape != self.initial_z.shape:
            raise ValueError(f"algebraic vector has shape {z.shape}, expected {self.initial_z.shape}")
        return {name: float(z[i]) for i, name in enumerate(self.variable_names)}

    def residual_vector(self, t: float, z: np.ndarray, inputs: Mapping[str, Any]) -> np.ndarray:
        evaluated = dict(self._evaluator(t, self.values_from_z(z), inputs))
        if self.strict_residuals:
            missing = [name for name in self.residual_names if name not in evaluated]
            extra = sorted(set(evaluated) - set(self.residual_names))
            if missing or extra:
                raise NetworkStructureError(
                    f"network residual evaluator for '{self.name}' returned incompatible residuals; "
                    f"missing={missing or 'none'}, extra={extra or 'none'}"
                )
        return np.array([float(evaluated.get(name, 0.0)) for name in self.residual_names], dtype=float)

    def solve(self, t: float, z0: np.ndarray | WarmStart | None, inputs: Mapping[str, Any]) -> NetworkSolution:
        return self.solve_limited(t, z0, inputs)

    def solve_limited(
        self,
        t: float,
        z0: np.ndarray | WarmStart | None,
        inputs: Mapping[str, Any],
        *,
        max_nfev: int | None = None,
    ) -> NetworkSolution:
        guess = self._initial_guess(z0)
        if self.n_algebraic == 0 and len(self.residual_definitions) == 0:
            solution = NetworkSolution(
                z=guess,
                values={},
                residuals={},
                normalized_residuals={},
                success=True,
                message="empty network solved",
            )
            if isinstance(z0, WarmStart):
                z0.update(solution)
            return solution

        def scaled_residual(z: np.ndarray) -> np.ndarray:
            return self.residual_vector(t, z, inputs) / self.residual_scales

        bounded = bool(np.any(np.isfinite(self.lower_bounds)) or np.any(np.isfinite(self.upper_bounds)))
        if bounded:
            success = False
            message = "bounded network; using least-squares solve"
            z = np.clip(guess, self.lower_bounds, self.upper_bounds)
        elif len(self.variables) == len(self.residual_definitions):
            root_result = root(scaled_residual, guess, method="hybr")
            success = bool(root_result.success)
            message = str(root_result.message)
            z = np.asarray(root_result.x, dtype=float)
        else:
            success = False
            message = "non-square network; using least-squares solve"
            z = guess

        if not success:
            ls_guess = np.clip(guess, self.lower_bounds, self.upper_bounds)
            ls_result = least_squares(
                scaled_residual,
                ls_guess,
                x_scale=self.variable_scales,
                jac_sparsity=self.sparse_jacobian_hint,
                bounds=(self.lower_bounds, self.upper_bounds),
                max_nfev=max_nfev,
            )
            z = np.asarray(ls_result.x, dtype=float)
            success = bool(ls_result.success)
            message = str(ls_result.message)

        residual_vector = self.residual_vector(t, z, inputs)
        residuals = {name: float(residual_vector[i]) for i, name in enumerate(self.residual_names)}
        normalized = {
            name: float(residual_vector[i] / self.residual_scales[i])
            for i, name in enumerate(self.residual_names)
        }
        solution = NetworkSolution(
            z=z,
            values=self.values_from_z(z),
            residuals=residuals,
            normalized_residuals=normalized,
            success=success,
            message=message,
        )
        if isinstance(z0, WarmStart):
            z0.update(solution)
        return solution

    def solve_checked(
        self,
        t: float,
        z0: np.ndarray | WarmStart | None,
        inputs: Mapping[str, Any],
        *,
        residual_tolerance: float = 1.0e-8,
        max_nfev: int | None = None,
    ) -> NetworkSolution:
        solution = self.solve_limited(t, z0, inputs, max_nfev=max_nfev)
        _, residual = solution.max_normalized_residual
        if not solution.success or abs(residual) > residual_tolerance:
            raise NetworkSolveError(self.name, solution, residual_tolerance)
        return solution

    def trim(
        self,
        inputs: Mapping[str, Any],
        *,
        z0: np.ndarray | WarmStart | None = None,
        t: float = 0.0,
        residual_tolerance: float = 1.0e-8,
    ) -> NetworkSolution:
        """Solve a steady algebraic operating point using the same residuals."""

        return self.solve_checked(t, z0, inputs, residual_tolerance=residual_tolerance)

    def _initial_guess(self, z0: np.ndarray | WarmStart | None) -> np.ndarray:
        if isinstance(z0, WarmStart):
            guess = z0.z
        else:
            guess = self.initial_z if z0 is None else z0
        guess = np.asarray(guess, dtype=float)
        if guess.shape != self.initial_z.shape:
            raise ValueError(f"algebraic initial guess has shape {guess.shape}, expected {self.initial_z.shape}")
        return guess

    def _validate_structure(self) -> None:
        variable_duplicates = _duplicates(self.variable_names)
        residual_duplicates = _duplicates(self.residual_names)
        if variable_duplicates:
            raise NetworkStructureError(f"duplicate algebraic variables in '{self.name}': {variable_duplicates}")
        if residual_duplicates:
            raise NetworkStructureError(f"duplicate algebraic residuals in '{self.name}': {residual_duplicates}")
        if self.require_square and len(self.variables) != len(self.residual_definitions):
            raise NetworkStructureError(
                f"network problem '{self.name}' must be square; got "
                f"{len(self.variables)} algebraic variables and {len(self.residual_definitions)} residuals"
            )


def _duplicates(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates
