from __future__ import annotations

from typing import Any, Dict, Mapping


def evaluate_controllers(config, targets: Mapping[str, Any], timings: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Evaluate configured controller blocks.

    This is intentionally small, but it centralizes the controller semantics
    that were previously duplicated in examples. It is the first slice of a
    ROCETS-like Run Processor for command generation.
    """

    if config is None:
        return {}
    timings = timings or {}
    outputs: Dict[str, Any] = {}
    for name, controller in config.controllers.items():
        controller_type = controller.get("type")
        if controller_type is None or controller_type == "null":
            outputs[str(controller["output"])] = _lookup_signal(str(controller["input"]), targets, timings, outputs)
        elif controller_type == "of_mass_flow_split":
            inputs = controller["inputs"]
            split_outputs = controller["outputs"]
            mdot_total = float(_lookup_signal(str(inputs["mdot_total"]), targets, timings, outputs))
            of_ratio = float(_lookup_signal(str(inputs["OF"]), targets, timings, outputs))
            mdot_fuel = mdot_total / (1.0 + of_ratio)
            outputs[str(split_outputs["oxidizer"])] = mdot_total - mdot_fuel
            outputs[str(split_outputs["fuel"])] = mdot_fuel
        elif controller_type == "gain_product":
            inputs = controller["inputs"]
            value = float(_lookup_signal(str(inputs["value"]), targets, timings, outputs))
            gain = float(_lookup_signal(str(inputs["gain"]), targets, timings, outputs))
            outputs[str(controller["output"])] = value * gain
        else:
            raise ValueError(f"Unsupported controller '{name}' type: {controller.get('type')}")
    return outputs


def _lookup_signal(path: str, targets: Mapping[str, Any], timings: Mapping[str, Any], commands: Mapping[str, Any]) -> Any:
    if path.startswith("targets."):
        return _lookup_nested(targets, path[len("targets."):])
    if path.startswith("timings."):
        return timings[path[len("timings."):]]
    if path in commands:
        return commands[path]
    raise ValueError(f"Unknown controller signal: {path}")


def _lookup_nested(data: Mapping[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        value = value[part]
    return value
