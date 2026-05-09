from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from atha.config.schema import ConfigError
from atha.config.transients import TransientBlock
from atha.core.component import BaseComponent


def transient_component_name(block: TransientBlock) -> str:
    """Return the component prefix used for a transient output path."""
    output_path = block.output_path
    if "." not in output_path:
        return block.name
    return output_path.rsplit(".", 1)[0]


def transient_state_names(block: TransientBlock) -> list[str]:
    """Return local state names that preserve output-path readability."""
    if block.type == "table":
        return []
    leaf = block.output_path.rsplit(".", 1)[-1]
    if block.type == "second_order":
        return [leaf, f"{leaf}_rate"]
    if block.type in {"first_order", "linear", "rate_limited"}:
        return [leaf]
    raise ConfigError(f"Unsupported transient type '{block.type}' for '{block.name}'")


class TransientBlockComponent(BaseComponent):
    """Engine component adapter for a scalar transient block."""

    def __init__(self, block: TransientBlock, command_values: Mapping[str, float] | None = None) -> None:
        self.block = block
        self._initial_state = block.initial_state(command_values)
        self._local_state_names = transient_state_names(block)
        super().__init__(transient_component_name(block))

    def _declare_ports(self) -> None:
        return None

    def _declare_states(self) -> None:
        for i, name in enumerate(self._local_state_names):
            self._register_state(name, float(self._initial_state[i]))

    def _declare_algebraic_vars(self) -> None:
        return None

    def compute_outputs(self, t: float, states: Dict[str, float], inputs: Dict[str, float]) -> Dict[str, float]:
        state_vector = self._state_vector(states)
        value = self.block.output(t, state_vector, inputs)
        outputs = {
            "output": value,
            self.block.output_path.rsplit(".", 1)[-1]: value,
            self.block.output_path: value,
            f"{self.block.name}.output": value,
        }
        if self.block.type == "second_order" and len(state_vector) == 2:
            outputs["rate"] = float(state_vector[1])
            outputs[f"{self.block.output_path}_rate"] = float(state_vector[1])
        return outputs

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        _ = outputs
        derivative = self.block.derivative(t, self._state_vector(states), inputs)
        return {name: float(derivative[i]) for i, name in enumerate(self._local_state_names)}

    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        _ = (t, states, inputs, outputs)
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        initial = self.block.initial_state(operating_point)
        for i, name in enumerate(self._local_state_names):
            self._state_values[name] = float(initial[i])

    def _state_vector(self, states: Dict[str, float]) -> np.ndarray:
        if not self._local_state_names:
            return np.zeros(0, dtype=float)
        return np.array([float(states[name]) for name in self._local_state_names], dtype=float)
