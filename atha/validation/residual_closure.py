from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from atha.components.registry import component_residual_contract
from atha.components.residuals import ResidualEvaluationContext
from atha.config.schema import ComponentConfig


@dataclass(frozen=True)
class ResidualClosureCheck:
    component: str
    residual: str
    value: float
    limit: float
    passed: bool


def evaluate_component_residual_closure(
    component: ComponentConfig,
    *,
    z: Mapping[str, float] | None = None,
    inputs: Mapping[str, float] | None = None,
    model: Mapping[str, Any] | None = None,
    limit: float = 1.0e-9,
) -> list[ResidualClosureCheck]:
    """Evaluate one component residual provider and return pass/fail checks."""

    contract = component_residual_contract(component)
    if contract is None:
        raise ValueError(f"component type {component.type!r} does not define a residual contract")
    residuals = contract.evaluate(
        component,
        ResidualEvaluationContext(
            z=dict(z or {}),
            inputs=dict(inputs or {}),
            model=dict(model or {}),
        ),
    )
    residual_names = {definition.name for definition in contract.residuals(component, dict(model or {}))}
    return [
        ResidualClosureCheck(
            component=component.name,
            residual=name,
            value=float(value),
            limit=float(limit),
            passed=bool(np.isfinite(value) and abs(float(value)) <= limit),
        )
        for name, value in residuals.items()
        if name in residual_names
    ]


def assert_component_residual_closure(
    component: ComponentConfig,
    *,
    z: Mapping[str, float] | None = None,
    inputs: Mapping[str, float] | None = None,
    model: Mapping[str, Any] | None = None,
    limit: float = 1.0e-9,
) -> list[ResidualClosureCheck]:
    """Evaluate residual closure and raise with residual names on failure."""

    checks = evaluate_component_residual_closure(
        component,
        z=z,
        inputs=inputs,
        model=model,
        limit=limit,
    )
    failed = [check for check in checks if not check.passed]
    if failed:
        details = ", ".join(f"{check.residual}={check.value:.6g}" for check in failed)
        raise AssertionError(f"component residual closure failed for {component.name}: {details}")
    return checks
