from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np

from atha.config.schedules import evaluate_schedule
from atha.config.schema import ConfigError, TransientConfig


SUPPORTED_TRANSIENT_TYPES = {"table", "first_order", "second_order", "linear", "rate_limited"}


@dataclass(frozen=True)
class TransientBlock:
    """Runtime model for a commanded scalar transient response."""

    config: TransientConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def type(self) -> str:
        return self.config.type

    @property
    def command_path(self) -> str:
        path = self.config.command.get("path", self.config.parameters.get("input"))
        if not isinstance(path, str) or not path:
            raise ConfigError(f"transient '{self.name}' requires command.path or input")
        return path

    @property
    def output_path(self) -> str:
        path = self.config.state.get("path", self.config.parameters.get("output"))
        if not isinstance(path, str) or not path:
            raise ConfigError(f"transient '{self.name}' requires state.path or output")
        return path

    @property
    def n_states(self) -> int:
        if self.type == "table":
            return 0
        if self.type == "second_order":
            return 2
        if self.type in {"first_order", "linear", "rate_limited"}:
            return 1
        raise ConfigError(f"Unsupported transient type '{self.type}' for '{self.name}'")

    def initial_state(self, command_values: Mapping[str, float] | None = None) -> np.ndarray:
        command_values = command_values or {}
        initial = float(self.config.state.get("initial", self.config.parameters.get("initial", command_values.get(self.command_path, 0.0))))
        if self.type == "table":
            return np.zeros(0)
        if self.type == "second_order":
            velocity = float(self.config.state.get("initial_rate", self.config.parameters.get("initial_rate", 0.0)))
            return np.array([initial, velocity], dtype=float)
        return np.array([initial], dtype=float)

    def output(self, t: float, state: np.ndarray, command_values: Mapping[str, float]) -> float:
        if self.type == "table":
            schedule = self.config.parameters.get("schedule", self.config.parameters)
            return float(evaluate_schedule(schedule, t, base_path=self.config.path))
        if state.size == 0:
            return float(command_values.get(self.command_path, 0.0))
        return self._clamp(float(state[0]))

    def derivative(self, _t: float, state: np.ndarray, command_values: Mapping[str, float]) -> np.ndarray:
        if self.type == "table":
            return np.zeros(0)

        command = self._clamp(float(command_values.get(self.command_path, state[0])))
        current = self._clamp(float(state[0]))
        error = command - current

        if self.type == "first_order":
            tau = max(float(self.config.parameters.get("time_constant", self.config.parameters.get("tau", 0.1))), 1.0e-12)
            return np.array([error / tau], dtype=float)

        if self.type == "second_order":
            velocity = float(state[1])
            omega_n = self._natural_frequency_rad_s()
            zeta = float(self.config.parameters.get("damping_ratio", self.config.parameters.get("zeta", 1.0)))
            acceleration = omega_n * omega_n * error - 2.0 * zeta * omega_n * velocity
            return np.array([velocity, acceleration], dtype=float)

        if self.type == "linear":
            rate = self._linear_rate(current, command)
            if abs(error) <= max(rate * 1.0e-6, 1.0e-12):
                return np.array([0.0], dtype=float)
            return np.array([math.copysign(rate, error)], dtype=float)

        if self.type == "rate_limited":
            opening_rate = float(self.config.parameters.get("opening_rate", self.config.parameters.get("rate", 1.0)))
            closing_rate = float(self.config.parameters.get("closing_rate", self.config.parameters.get("rate", opening_rate)))
            rate = opening_rate if error >= 0.0 else closing_rate
            return np.array([_signed_limit(error, rate)], dtype=float)

        raise ConfigError(f"Unsupported transient type '{self.type}' for '{self.name}'")

    def _linear_rate(self, current: float, command: float) -> float:
        if "rate" in self.config.parameters:
            return abs(float(self.config.parameters["rate"]))
        duration = max(float(self.config.parameters.get("duration", 1.0)), 1.0e-12)
        lower = float(self.config.parameters.get("lower_limit", self.config.state.get("lower_limit", 0.0)))
        upper = float(self.config.parameters.get("upper_limit", self.config.state.get("upper_limit", 1.0)))
        full_span = abs(float(self.config.parameters.get("span", upper - lower)))
        if "from" in self.config.parameters and "to" in self.config.parameters:
            full_span = abs(float(self.config.parameters["to"]) - float(self.config.parameters["from"]))
        return max(full_span / duration, 0.0)

    def _natural_frequency_rad_s(self) -> float:
        if "natural_frequency_hz" in self.config.parameters:
            return 2.0 * math.pi * float(self.config.parameters["natural_frequency_hz"])
        return float(self.config.parameters.get("natural_frequency_rad_s", self.config.parameters.get("omega_n", 1.0)))

    def _clamp(self, value: float) -> float:
        lower = self.config.parameters.get("lower_limit", self.config.state.get("lower_limit"))
        upper = self.config.parameters.get("upper_limit", self.config.state.get("upper_limit"))
        if lower is not None:
            value = max(float(lower), value)
        if upper is not None:
            value = min(float(upper), value)
        return value


class TransientSystem:
    """Collection of scalar transient blocks with one flat state vector."""

    def __init__(self, blocks: Iterable[TransientBlock]) -> None:
        self.blocks = list(blocks)
        self._slices: Dict[str, slice] = {}
        start = 0
        for block in self.blocks:
            stop = start + block.n_states
            self._slices[block.name] = slice(start, stop)
            start = stop
        self.n_states = start

    @classmethod
    def from_configs(cls, configs: Mapping[str, TransientConfig]) -> "TransientSystem":
        blocks = []
        for config in configs.values():
            if config.type not in SUPPORTED_TRANSIENT_TYPES:
                raise ConfigError(
                    f"Unsupported transient type '{config.type}' for '{config.name}'. "
                    f"Supported types: {sorted(SUPPORTED_TRANSIENT_TYPES)}"
                )
            blocks.append(TransientBlock(config))
        return cls(blocks)

    def initial_state(self, command_values: Mapping[str, float] | None = None) -> np.ndarray:
        if not self.blocks:
            return np.zeros(0)
        return np.concatenate([block.initial_state(command_values) for block in self.blocks])

    def evaluate(self, t: float, state: np.ndarray, command_values: Mapping[str, float]) -> Dict[str, float]:
        outputs: Dict[str, float] = {}
        for block in self.blocks:
            local = state[self._slices[block.name]]
            value = block.output(t, local, command_values)
            outputs[block.output_path] = value
            outputs[f"{block.name}.output"] = value
        return outputs

    def derivatives(self, t: float, state: np.ndarray, command_values: Mapping[str, float]) -> np.ndarray:
        if not self.blocks:
            return np.zeros(0)
        pieces = []
        for block in self.blocks:
            local = state[self._slices[block.name]]
            pieces.append(block.derivative(t, local, command_values))
        return np.concatenate(pieces) if pieces else np.zeros(0)


def _signed_limit(error: float, rate: float) -> float:
    rate = abs(float(rate))
    if abs(error) <= 1.0e-12:
        return 0.0
    return math.copysign(min(abs(error), rate), error)
