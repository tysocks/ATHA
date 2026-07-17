from __future__ import annotations

import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


_FUNCTION_CACHE: dict[tuple[Path, str], Any] = {}


@dataclass(frozen=True)
class ControllerStateInfo:
    name: str
    initial: float = 0.0


STATEFUL_CONTROLLER_TYPES = {"pi", "pid"}


def evaluate_controllers(
    config,
    targets: Mapping[str, Any],
    timings: Mapping[str, Any] | None = None,
    measurements: Mapping[str, Any] | None = None,
    *,
    dt: float = 1.0,
    current_phase: str | None = None,
    previous_outputs: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate configured controller blocks.

    This is intentionally small, but it centralizes the controller semantics
    that were previously duplicated in examples. It is the first slice of a
    ROCETS-like Run Processor for command generation.
    """

    if config is None:
        return {}
    timings = timings or {}
    measurements = measurements or {}
    previous_outputs = previous_outputs or {}
    outputs: Dict[str, Any] = {}
    for name in controller_execution_order(config.controllers):
        controller = config.controllers[name]
        if not controller_is_active(controller, current_phase):
            continue
        controller_type = controller.get("type")
        if controller_type is None or controller_type == "null":
            outputs[str(controller["output"])] = _lookup_signal(str(controller["input"]), targets, timings, measurements, outputs)
        elif controller_type == "of_mass_flow_split":
            inputs = controller["inputs"]
            split_outputs = controller["outputs"]
            mdot_total = float(_lookup_signal(str(inputs["mdot_total"]), targets, timings, measurements, outputs))
            of_ratio = float(_lookup_signal(str(inputs["OF"]), targets, timings, measurements, outputs))
            mdot_fuel = mdot_total / (1.0 + of_ratio)
            outputs[str(split_outputs["oxidizer"])] = mdot_total - mdot_fuel
            outputs[str(split_outputs["fuel"])] = mdot_fuel
        elif controller_type == "gain_product":
            inputs = controller["inputs"]
            value = float(_lookup_signal(str(inputs["value"]), targets, timings, measurements, outputs))
            gain = float(_lookup_signal(str(inputs["gain"]), targets, timings, measurements, outputs))
            outputs[str(controller["output"])] = value * gain
        elif controller_type == "proportional":
            outputs.update(_evaluate_feedback_controller(name, controller, targets, timings, measurements, outputs, dt=dt, previous_outputs=previous_outputs))
        elif controller_type in {"pi", "pid"}:
            outputs.update(_evaluate_feedback_controller(name, controller, targets, timings, measurements, outputs, dt=dt, previous_outputs=previous_outputs))
        elif controller_type == "scheduled_gain":
            inputs = controller["inputs"]
            value = float(_lookup_signal(str(inputs["value"]), targets, timings, measurements, outputs))
            gain = float(_lookup_signal(str(inputs["gain"]), targets, timings, measurements, outputs))
            outputs[str(controller["output"])] = value * gain
        elif controller_type == "limiter":
            params = controller.get("parameters", {})
            value = float(_lookup_signal(str(controller["input"]), targets, timings, measurements, outputs))
            lower = float(params.get("lower_limit", params.get("min", -float("inf"))))
            upper = float(params.get("upper_limit", params.get("max", float("inf"))))
            limited = min(max(value, lower), upper)
            outputs[str(controller["output"])] = limited
            outputs[f"controller.{name}.command"] = limited
            outputs[f"controller.{name}.saturated"] = float(limited != value)
        elif controller_type == "rate_limiter":
            params = controller.get("parameters", {})
            value = float(_lookup_signal(str(controller["input"]), targets, timings, measurements, outputs))
            previous = float(params.get("initial", value))
            # Stateless compatibility path: without driver-owned controller
            # states, report the requested value and expose diagnostics. The
            # stateful DAE path will use controller_state_infos() for memory.
            outputs[str(controller["output"])] = value
            outputs[f"controller.{name}.command"] = value
            outputs[f"controller.{name}.rate"] = value - previous
        elif controller_type in {"selector", "min", "max"}:
            input_paths = controller.get("inputs", [])
            if isinstance(input_paths, Mapping):
                input_paths = list(input_paths.values())
            values = [float(_lookup_signal(str(path), targets, timings, measurements, outputs)) for path in input_paths]
            mode = str(controller.get("mode", controller_type))
            if mode in {"min", "selector_min"} or controller_type == "min":
                selected = min(values)
            elif mode in {"max", "selector_max"} or controller_type == "max":
                selected = max(values)
            else:
                index = int(controller.get("index", 0))
                selected = values[index]
            outputs[str(controller["output"])] = selected
            outputs[f"controller.{name}.command"] = selected
        else:
            raise ValueError(f"Unsupported controller '{name}' type: {controller.get('type')}")
    return outputs


def evaluate_dynamic_controllers(
    config,
    targets: Mapping[str, Any],
    timings: Mapping[str, Any] | None = None,
    measurements: Mapping[str, Any] | None = None,
    *,
    dt: float = 1.0,
    current_phase: str | None = None,
    previous_outputs: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate controllers that may depend on live simulation measurements."""

    if config is None:
        return {}
    timings = timings or {}
    measurements = measurements or {}
    previous_outputs = previous_outputs or {}
    outputs: Dict[str, Any] = {}
    for name in controller_execution_order(config.controllers):
        controller = config.controllers[name]
        if not controller_is_active(controller, current_phase):
            continue
        controller_type = controller.get("type")
        if controller_type == "python_function":
            function = _load_function(controller, config.path)
            result = function(
                targets=targets,
                timings=timings,
                measurements=measurements,
                commands=outputs,
                parameters=controller.get("parameters", {}),
            )
            if not isinstance(result, Mapping):
                raise ValueError(f"Controller '{name}' function must return a mapping")
            output_map = controller.get("outputs")
            if isinstance(output_map, Mapping):
                for alias, path in output_map.items():
                    if alias not in result:
                        raise ValueError(f"Controller '{name}' result missing output '{alias}'")
                    outputs[str(path)] = result[alias]
            else:
                outputs.update({str(key): value for key, value in result.items()})
        else:
            sub_config = type("_SingleControllerConfig", (), {"controllers": {name: controller}})()
            outputs.update(
                evaluate_controllers(
                    sub_config,
                    targets,
                    {**timings, **outputs},
                    measurements,
                    dt=dt,
                    current_phase=current_phase,
                    previous_outputs=previous_outputs,
                )
            )
    return outputs


def controller_evaluation_period(config) -> float | None:
    """Return the configured controller sample period in seconds, if any."""

    if config is None:
        return None
    evaluation = getattr(config, "evaluation", {}) or {}
    if not isinstance(evaluation, Mapping):
        return None
    if "period_s" in evaluation:
        period = float(evaluation["period_s"])
    elif "frequency_hz" in evaluation:
        frequency = float(evaluation["frequency_hz"])
        period = 1.0 / frequency if frequency > 0.0 else 0.0
    else:
        return None
    if period <= 0.0:
        raise ValueError("controllers.evaluation period/frequency must be positive")
    return period


def controller_sample_index(t: float, period_s: float | None) -> int | None:
    if period_s is None:
        return None
    return int(np.floor((float(t) + 1.0e-12) / period_s))


def controller_state_infos(config) -> list[ControllerStateInfo]:
    """Return controller states that must join the global state vector."""

    if config is None:
        return []
    states: list[ControllerStateInfo] = []
    for name, controller in config.controllers.items():
        controller_type = str(controller.get("type", "null"))
        params = controller.get("parameters", {})
        if controller_type in STATEFUL_CONTROLLER_TYPES:
            states.append(ControllerStateInfo(f"controller.{name}.integral", float(params.get("integral_initial", 0.0))))
        if controller_type == "rate_limiter":
            states.append(ControllerStateInfo(f"controller.{name}.previous_command", float(params.get("initial", 0.0))))
    return states


def controller_is_active(controller: Mapping[str, Any], current_phase: str | None) -> bool:
    from atha.config.mission_phases import controller_is_active as _phase_active

    return _phase_active(controller, current_phase)


def controller_reset_state_values(controller_name: str, controller: Mapping[str, Any]) -> dict[str, float]:
    """Return default memory values used when resetting a controller on phase entry."""

    params = controller.get("parameters", {})
    values: dict[str, float] = {}
    controller_type = str(controller.get("type", "null"))
    if controller_type in STATEFUL_CONTROLLER_TYPES:
        values[f"controller.{controller_name}.integral"] = float(params.get("integral_initial", 0.0))
    if controller_type == "rate_limiter":
        values[f"controller.{controller_name}.previous_command"] = float(params.get("initial", 0.0))
    return values


def controller_execution_order(controllers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Topologically order controllers using command dependencies."""

    names = list(controllers)
    producers: dict[str, str] = {}
    for name, controller in controllers.items():
        for output in controller_output_paths(controller):
            producers[output] = name

    dependencies = {
        name: {
            producers[path]
            for path in controller_input_paths(controller)
            if path in producers and producers[path] != name
        }
        for name, controller in controllers.items()
    }
    ordered: list[str] = []
    pending = set(names)
    while pending:
        ready = sorted(name for name in pending if not dependencies[name] - set(ordered))
        if not ready:
            cycle = ", ".join(sorted(pending))
            raise ValueError(f"Controller dependency cycle detected among: {cycle}")
        ordered.extend(ready)
        pending -= set(ready)
    return ordered


def controller_output_paths(controller: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    if "output" in controller:
        paths.append(str(controller["output"]))
    output_map = controller.get("outputs")
    if isinstance(output_map, Mapping):
        paths.extend(str(path) for path in output_map.values())
    return paths


def controller_input_paths(controller: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    if "input" in controller:
        paths.append(str(controller["input"]))
    inputs = controller.get("inputs")
    if isinstance(inputs, Mapping):
        paths.extend(str(path) for path in inputs.values())
    elif isinstance(inputs, list):
        paths.extend(str(path) for path in inputs)
    return paths


def _lookup_signal(
    path: str,
    targets: Mapping[str, Any],
    timings: Mapping[str, Any],
    measurements: Mapping[str, Any],
    commands: Mapping[str, Any],
) -> Any:
    if path.startswith("targets."):
        return _lookup_nested(targets, path[len("targets."):])
    if path.startswith("timings."):
        return timings[path[len("timings."):]]
    if path.startswith("measurements."):
        key = path[len("measurements.") :]
        if key in measurements:
            return measurements[key]
        dotted = key.replace("_", ".")
        if dotted in measurements:
            return measurements[dotted]
        raise ValueError(f"Unknown controller signal: {path}")
    if path in commands:
        return commands[path]
    if path in targets:
        return targets[path]
    if path in timings:
        return timings[path]
    raise ValueError(f"Unknown controller signal: {path}")


def _evaluate_feedback_controller(
    name: str,
    controller: Mapping[str, Any],
    targets: Mapping[str, Any],
    timings: Mapping[str, Any],
    measurements: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    dt: float = 1.0,
    previous_outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    controller_type = str(controller.get("type", "proportional"))
    inputs = controller["inputs"]
    params = controller.get("parameters", {})
    target = float(_lookup_signal(str(inputs["target"]), targets, timings, measurements, outputs))
    measurement = float(_lookup_signal(str(inputs["measurement"]), targets, timings, measurements, outputs))
    error = target - measurement
    bias = float(params.get("bias", 0.0))
    feed_forward = float(params.get("feed_forward_gain", 0.0)) * target
    kp = float(params.get("gain", params.get("proportional_gain", params.get("kp", 0.0))))
    ki = float(params.get("ki", params.get("integral_gain", 0.0))) if controller_type in {"pi", "pid"} else 0.0
    kd = float(params.get("kd", params.get("derivative_gain", 0.0))) if controller_type == "pid" else 0.0
    previous_outputs = previous_outputs or {}
    previous_integral = float(previous_outputs.get(f"controller.{name}.integral", params.get("integral_initial", 0.0)))
    previous_saturated = bool(float(previous_outputs.get(f"controller.{name}.saturated", 0.0)))
    anti_windup = bool(params.get("anti_windup", True))
    if controller_type in {"pi", "pid"} and not (anti_windup and previous_saturated):
        integral = previous_integral + error * max(float(dt), 0.0)
    else:
        integral = previous_integral
    previous_error = float(previous_outputs.get(f"controller.{name}.error", params.get("previous_error_initial", error)))
    derivative = (error - previous_error) / max(float(dt), 1.0e-12) if controller_type == "pid" else 0.0
    proportional_term = kp * error
    integral_term = ki * integral
    derivative_term = kd * derivative
    raw = bias + feed_forward + proportional_term + integral_term + derivative_term
    lower = float(params.get("lower_limit", params.get("min", -float("inf"))))
    upper = float(params.get("upper_limit", params.get("max", float("inf"))))
    command = min(max(raw, lower), upper)
    return {
        str(controller["output"]): command,
        f"controller.{name}.target": target,
        f"controller.{name}.measurement": measurement,
        f"controller.{name}.error": error,
        f"controller.{name}.command": command,
        f"controller.{name}.raw_command": raw,
        f"controller.{name}.saturated": float(command != raw),
        f"controller.{name}.integral": integral,
        f"controller.{name}.derivative": derivative,
        f"controller.{name}.proportional_term": proportional_term,
        f"controller.{name}.integral_term": integral_term,
        f"controller.{name}.derivative_term": derivative_term,
    }


def _lookup_nested(data: Mapping[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        value = value[part]
    return value


def _load_function(controller: Mapping[str, Any], config_path: Path | None):
    function_spec = controller.get("function", {})
    if not isinstance(function_spec, Mapping):
        raise ValueError("python_function controller requires function mapping")
    path_value = function_spec.get("path")
    name = str(function_spec.get("name", "controller"))
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("python_function controller requires function.path")
    path = Path(path_value)
    if not path.is_absolute():
        if config_path is None:
            path = path.resolve()
        else:
            path = (config_path.parent / path).resolve()
    cache_key = (path, name)
    if cache_key in _FUNCTION_CACHE:
        return _FUNCTION_CACHE[cache_key]
    spec = importlib.util.spec_from_file_location(f"atha_user_controller_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load controller function from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        function = getattr(module, name)
    except AttributeError as exc:
        raise ValueError(f"Controller function '{name}' not found in {path}") from exc
    _FUNCTION_CACHE[cache_key] = function
    return function
