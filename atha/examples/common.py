from __future__ import annotations

from typing import Any

from atha.config import evaluate_boundary_conditions
from atha.solver.steady_state import _evaluate_pass, _seed_from_states


def coerce_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: coerce_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [coerce_numbers(v) for v in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def component_params(loaded, name: str) -> dict:
    return coerce_numbers(dict(loaded.engine.components[name].parameters))


def boundary_values(loaded, t: float = 0.0) -> dict:
    if loaded.boundary_conditions is None:
        return {}
    return coerce_numbers(evaluate_boundary_conditions(loaded.boundary_conditions, t))


def connect_from_config(engine, connections) -> None:
    for conn in connections:
        src_comp, src_port = conn.source.split(".", 1)
        dst_comp, dst_port = conn.target.split(".", 1)
        engine.connect(engine[src_comp].port(src_port), engine[dst_comp].port(dst_port))


def evaluate_outputs(layout, X, bcs: dict) -> dict:
    layout.scatter_state_vector(X)
    context = {}
    _seed_from_states(layout, context, bcs)
    context.update(bcs)
    _evaluate_pass(layout, X, context, bcs, collect_residuals=False)
    _evaluate_pass(layout, X, context, bcs, collect_residuals=False)
    return {comp.name: dict(comp.last_outputs) for comp in layout.components}

