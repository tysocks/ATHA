from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from atha.config.schema import ConfigError
from atha.network import NetworkProblem, NetworkResidual, NetworkVariable


_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
}


@dataclass(frozen=True)
class BalanceResidual:
    expression: str
    scale: float = 1.0


@dataclass(frozen=True)
class BalanceConfig:
    name: str
    residual: BalanceResidual
    variable: str
    initial: float = 0.0
    bounds: tuple[float | None, float | None] = (None, None)
    units: str = ""


def balance_configs(raw: Any) -> list[BalanceConfig]:
    if raw in (None, {}):
        return []
    if not isinstance(raw, Mapping):
        raise ConfigError("analysis.balances must be a mapping")
    balances: list[BalanceConfig] = []
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"analysis.balances.{name} must be a mapping")
        residual = spec.get("residual")
        if not isinstance(residual, Mapping):
            raise ConfigError(f"analysis.balances.{name}.residual must be a mapping")
        expression = residual.get("expression")
        if not isinstance(expression, str) or not expression:
            raise ConfigError(f"analysis.balances.{name}.residual.expression must be a non-empty string")
        variable = spec.get("variable")
        if not isinstance(variable, str) or not variable:
            raise ConfigError(f"analysis.balances.{name}.variable must be a non-empty source path")
        bounds = _bounds(spec.get("bounds"))
        balances.append(
            BalanceConfig(
                name=str(name),
                residual=BalanceResidual(expression=expression, scale=float(residual.get("scale", 1.0))),
                variable=variable,
                initial=float(spec.get("initial", 0.5 * (bounds[0] + bounds[1]) if bounds[0] is not None and bounds[1] is not None else 0.0)),
                bounds=bounds,
                units=str(spec.get("units", "")),
            )
        )
    return balances


def wrap_problem_with_balances(
    problem: NetworkProblem,
    balances: list[BalanceConfig],
    *,
    allowed_sources: set[str],
) -> NetworkProblem:
    if not balances:
        return problem
    validators = [_ExpressionValidator(balance.residual.expression, allowed_sources | {balance.variable}) for balance in balances]
    variables = list(problem.variables)
    residuals = list(problem.residual_definitions)
    existing_variables = set(problem.variable_names)
    for balance in balances:
        if balance.variable not in existing_variables:
            variables.append(
                NetworkVariable(
                    balance.variable,
                    units=balance.units,
                    scale=max(abs(balance.initial), 1.0),
                    initial=balance.initial,
                    owner=f"balance.{balance.name}",
                    description=f"Trim variable for balance {balance.name}",
                    lower=balance.bounds[0],
                    upper=balance.bounds[1],
                )
            )
            existing_variables.add(balance.variable)
        residuals.append(
            NetworkResidual(
                f"balances.{balance.name}.residual",
                scale=balance.residual.scale,
                owner=f"balance.{balance.name}",
                description=balance.residual.expression,
            )
        )

    def evaluate(t: float, z: Mapping[str, float], inputs: Mapping[str, Any]) -> dict[str, float]:
        merged_inputs = _merge_inputs(inputs, z)
        base_z = {name: float(z[name]) for name in problem.variable_names}
        result = dict(problem._evaluator(t, base_z, merged_inputs))  # type: ignore[attr-defined]
        values = _catalog_values(merged_inputs, z, result)
        for balance, validator in zip(balances, validators):
            result[f"balances.{balance.name}.residual"] = validator.evaluate(values)
        return result

    return NetworkProblem(
        variables,
        residuals,
        evaluate,
        name=f"{problem.name}_balances",
        strict_residuals=problem.strict_residuals,
        require_square=problem.require_square,
    )


class _ExpressionValidator:
    def __init__(self, expression: str, allowed_sources: set[str]) -> None:
        self.expression = expression
        self.allowed_sources = allowed_sources
        self.tree = ast.parse(expression, mode="eval")
        self.paths = self._validate(self.tree)
        root_names = {path.split(".", 1)[0] for path in allowed_sources if "." in path}
        unknown = sorted(path for path in self.paths if path not in allowed_sources and path not in root_names)
        if unknown:
            raise ConfigError(f"balance expression references unknown source path(s): {unknown}")

    def evaluate(self, values: Mapping[str, Any]) -> float:
        return float(self._eval(self.tree.body, values))

    def _validate(self, node: ast.AST) -> set[str]:
        paths: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Expression):
                continue
            if isinstance(child, ast.Constant):
                if not isinstance(child.value, (int, float)):
                    raise ConfigError("balance expressions may only contain numeric constants")
                continue
            if isinstance(child, ast.BinOp):
                if not isinstance(child.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    raise ConfigError("balance expressions only support +, -, *, /, and **")
                continue
            if isinstance(child, ast.UnaryOp):
                if not isinstance(child.op, (ast.UAdd, ast.USub)):
                    raise ConfigError("balance expressions only support unary +/-")
                continue
            if isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name) or child.func.id not in _ALLOWED_FUNCTIONS:
                    raise ConfigError(f"unsupported balance expression function: {ast.unparse(child.func)}")
                continue
            if isinstance(child, ast.Name):
                if child.id not in _ALLOWED_FUNCTIONS:
                    paths.add(child.id)
                continue
            if isinstance(child, ast.Attribute):
                paths.add(_path_from_attribute(child))
                continue
            if isinstance(child, (ast.Load, ast.operator, ast.unaryop)):
                continue
            raise ConfigError(f"unsupported balance expression syntax: {type(child).__name__}")
        return {path for path in paths if not any(other != path and other.startswith(f"{path}.") for other in paths)}

    def _eval(self, node: ast.AST, values: Mapping[str, Any]) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_FUNCTIONS:
                return _ALLOWED_FUNCTIONS[node.id]  # type: ignore[return-value]
            return _numeric(values, node.id)
        if isinstance(node, ast.Attribute):
            return _numeric(values, _path_from_attribute(node))
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, values)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, values)
            right = self._eval(node.right, values)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _ALLOWED_FUNCTIONS[node.func.id]
            return float(fn(*[self._eval(arg, values) for arg in node.args]))
        raise ConfigError(f"unsupported balance expression syntax: {type(node).__name__}")


def _path_from_attribute(node: ast.Attribute) -> str:
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise ConfigError("balance expression attributes must start from a source path name")
    parts.append(current.id)
    return ".".join(reversed(parts))


def _catalog_values(inputs: Mapping[str, Any], z: Mapping[str, float], residuals: Mapping[str, float]) -> dict[str, Any]:
    values = dict(inputs)
    values.update(z)
    values.update(residuals)
    values.update({f"residuals.{key}": value for key, value in residuals.items()})
    for key, value in inputs.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                values[f"{key}.{nested_key}"] = nested_value
                if key == "inputs":
                    values[str(nested_key)] = nested_value
    return values


def _merge_inputs(inputs: Mapping[str, Any], z: Mapping[str, float]) -> dict[str, Any]:
    merged = dict(inputs)
    merged.update({key: float(value) for key, value in z.items()})
    if "inputs" in merged and isinstance(merged["inputs"], Mapping):
        nested = dict(merged["inputs"])
        nested.update({key: float(value) for key, value in z.items()})
        merged["inputs"] = nested
    return merged


def _numeric(values: Mapping[str, Any], path: str) -> float:
    if path not in values:
        raise ConfigError(f"balance expression source path unavailable at evaluation: {path}")
    value = values[path]
    if not isinstance(value, (int, float, np.floating)):
        raise ConfigError(f"balance expression source path is not numeric: {path}")
    return float(value)


def _bounds(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return (None, None)
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError("balance bounds must be [lower, upper]")
    lower = None if value[0] is None else float(value[0])
    upper = None if value[1] is None else float(value[1])
    if lower is not None and upper is not None and upper < lower:
        raise ConfigError("balance upper bound must be greater than or equal to lower bound")
    return (lower, upper)
