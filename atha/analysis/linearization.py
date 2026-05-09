from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


VectorFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PerturbationConfig:
    state_default: float = 1.0e-6
    input_default: float = 1.0e-6
    minimum_absolute: float = 1.0e-9
    per_state: Mapping[str, float] | None = None
    per_input: Mapping[str, float] | None = None


@dataclass(frozen=True)
class StateSpaceLinearization:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    state_labels: list[str]
    input_labels: list[str]
    output_labels: list[str]
    operating_point: dict[str, dict[str, float]]
    perturbations: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "D": self.D.tolist(),
            "state_labels": self.state_labels,
            "input_labels": self.input_labels,
            "output_labels": self.output_labels,
            "operating_point": self.operating_point,
            "perturbations": self.perturbations,
        }


def finite_difference_state_space(
    dynamics: VectorFunction,
    outputs: VectorFunction,
    x0: np.ndarray,
    u0: np.ndarray,
    *,
    state_labels: list[str],
    input_labels: list[str],
    output_labels: list[str],
    perturbations: PerturbationConfig | None = None,
) -> StateSpaceLinearization:
    """Linearize ``xdot=f(x,u)``, ``y=g(x,u)`` with central finite differences."""

    perturbations = perturbations or PerturbationConfig()
    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    f0 = np.asarray(dynamics(x0, u0), dtype=float)
    y0 = np.asarray(outputs(x0, u0), dtype=float)
    if f0.shape != x0.shape:
        raise ValueError(f"dynamics returned shape {f0.shape}; expected {x0.shape}")
    if len(state_labels) != x0.size:
        raise ValueError("state_labels length must match state vector length")
    if len(input_labels) != u0.size:
        raise ValueError("input_labels length must match input vector length")
    if len(output_labels) != y0.size:
        raise ValueError("output_labels length must match output vector length")

    A = np.zeros((x0.size, x0.size), dtype=float)
    B = np.zeros((x0.size, u0.size), dtype=float)
    C = np.zeros((y0.size, x0.size), dtype=float)
    D = np.zeros((y0.size, u0.size), dtype=float)
    state_steps = [_step(x0[i], state_labels[i], perturbations.state_default, perturbations) for i in range(x0.size)]
    input_steps = [_step(u0[i], input_labels[i], perturbations.input_default, perturbations, input_step=True) for i in range(u0.size)]

    for i, step in enumerate(state_steps):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += step
        xm[i] -= step
        A[:, i] = (dynamics(xp, u0) - dynamics(xm, u0)) / (2.0 * step)
        C[:, i] = (outputs(xp, u0) - outputs(xm, u0)) / (2.0 * step)

    for i, step in enumerate(input_steps):
        up = u0.copy()
        um = u0.copy()
        up[i] += step
        um[i] -= step
        B[:, i] = (dynamics(x0, up) - dynamics(x0, um)) / (2.0 * step)
        D[:, i] = (outputs(x0, up) - outputs(x0, um)) / (2.0 * step)

    return StateSpaceLinearization(
        A=A,
        B=B,
        C=C,
        D=D,
        state_labels=state_labels,
        input_labels=input_labels,
        output_labels=output_labels,
        operating_point={
            "states": {label: float(value) for label, value in zip(state_labels, x0)},
            "inputs": {label: float(value) for label, value in zip(input_labels, u0)},
            "outputs": {label: float(value) for label, value in zip(output_labels, y0)},
            "derivatives": {label: float(value) for label, value in zip(state_labels, f0)},
        },
        perturbations={
            "states": {label: float(value) for label, value in zip(state_labels, state_steps)},
            "inputs": {label: float(value) for label, value in zip(input_labels, input_steps)},
        },
    )


def write_linearization_json(path: str | Path, linearization: StateSpaceLinearization) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(linearization.to_dict(), stream, indent=2)
    return path


def _step(
    value: float,
    label: str,
    default_relative: float,
    perturbations: PerturbationConfig,
    *,
    input_step: bool = False,
) -> float:
    overrides = perturbations.per_input if input_step else perturbations.per_state
    if overrides and label in overrides:
        return max(abs(float(overrides[label])), perturbations.minimum_absolute)
    return max(abs(float(value)) * default_relative, perturbations.minimum_absolute)
