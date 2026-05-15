from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from atha.assembly import EngineAssembler
from atha.config import (
    TransientSystem,
    controller_execution_order,
    evaluate_boundary_conditions,
    evaluate_operating_targets,
    evaluate_timing_events,
)
from atha.config.controllers import controller_state_infos
from atha.config.controllers import controller_evaluation_period, controller_is_active, controller_sample_index
from atha.config.loader import LoadedAnalysisConfig
from atha.network import NetworkProblem, WarmStart
from atha.runner.solver_driver import ExecutionPlan


@dataclass
class DAEPoint:
    time: float
    states: dict[str, float]
    algebraics: dict[str, float]
    residuals: dict[str, float]
    normalized_residuals: dict[str, float]
    commands: dict[str, float]
    targets: dict[str, Any]
    boundaries: dict[str, Any]
    timings: dict[str, Any]
    measurements: dict[str, float]


@dataclass
class DAEExecutionResult:
    time: np.ndarray
    state_names: list[str]
    algebraic_names: list[str]
    residual_names: list[str]
    X: np.ndarray
    Z: np.ndarray
    residual_history: dict[str, np.ndarray]
    command_history: dict[str, np.ndarray]
    target_history: dict[str, np.ndarray]
    boundary_history: dict[str, np.ndarray]
    measurement_history: dict[str, np.ndarray]
    points: list[DAEPoint] = field(default_factory=list)


class DAEExecutionProblem:
    """Universal DAE execution-loop foundation.

    This class owns the common `X/Z/Rz` bookkeeping that compatibility runners
    previously duplicated or skipped. It evaluates schedules, controllers,
    transient blocks, state modes, and the port algebraic network through one
    path. Physical derivatives are still supplied by component models in later
    phases; today, transient and controller-state derivatives are integrated and
    other states default to zero unless a caller supplies a derivative hook.
    """

    def __init__(
        self,
        loaded: LoadedAnalysisConfig,
        execution_plan: ExecutionPlan,
        *,
        network_problem: NetworkProblem | None = None,
    ) -> None:
        self.loaded = loaded
        self.execution_plan = execution_plan
        self.assembler = EngineAssembler(loaded)
        self.network_problem = network_problem or self.assembler.port_network_problem()
        initial_vectors = self.assembler.initial_vectors()
        self.state_names = initial_vectors.state_names
        self.X0 = np.asarray(initial_vectors.X, dtype=float)
        self.algebraic_names = self.network_problem.variable_names
        self.Z0 = self.network_problem.initial_z.copy()
        self.transient_system = TransientSystem.from_configs(loaded.transients)
        self._transient_state_indexes = {
            name: self.state_names.index(name)
            for name in self.transient_system.state_names()
            if name in self.state_names
        }
        self._controller_state_indexes = {
            state.name: self.state_names.index(state.name)
            for state in controller_state_infos(loaded.controllers)
            if state.name in self.state_names
        }
        self._controller_period_s = controller_evaluation_period(loaded.controllers)
        self._controller_hold_cache: dict[int, dict[str, Any]] = {}
        self._shaft_couplings = _shaft_couplings(loaded.engine.components, loaded.engine.connections)

    def initial_state(self) -> np.ndarray:
        return self.X0.copy()

    def evaluate(self, t: float, x: np.ndarray, warm_start: WarmStart | np.ndarray | None = None) -> DAEPoint:
        states = self._state_dict(x)
        targets = evaluate_operating_targets(self.loaded.operating_conditions, t) if self.loaded.operating_conditions is not None else {}
        boundaries = evaluate_boundary_conditions(self.loaded.boundary_conditions, t) if self.loaded.boundary_conditions is not None else {}
        timings = evaluate_timing_events(self.loaded.timings, t)
        transient_state = self._transient_state_vector(x)
        commands = self._evaluate_controllers(t, targets, timings, {}, states)
        transient_sources = self.transient_system.sample_sources(t, transient_state, {**timings, **commands})
        network_inputs = self._network_inputs(states, targets, boundaries, timings, commands, transient_sources)
        solution = self.network_problem.solve(t, warm_start, {"inputs": network_inputs})
        measurements = self._measurements(solution.values, states)
        commands = self._evaluate_controllers(t, targets, timings, measurements, states)
        transient_sources = self.transient_system.sample_sources(t, transient_state, {**timings, **commands})
        network_inputs = self._network_inputs(states, targets, boundaries, timings, commands, transient_sources)
        solution = self.network_problem.solve(t, warm_start, {"inputs": network_inputs})
        measurements = self._measurements(solution.values, states)
        return DAEPoint(
            time=float(t),
            states=states,
            algebraics=solution.values,
            residuals=solution.residuals,
            normalized_residuals=solution.normalized_residuals,
            commands={key: float(value) for key, value in commands.items() if _is_number(value)},
            targets=dict(targets),
            boundaries=dict(boundaries),
            timings=dict(timings),
            measurements=measurements,
        )

    def rhs(self, t: float, x: np.ndarray, warm_start: WarmStart | None = None) -> np.ndarray:
        point = self.evaluate(t, x, warm_start)
        dx = np.zeros_like(x, dtype=float)
        transient_state = self._transient_state_vector(x)
        transient_derivatives = self.transient_system.derivatives(t, transient_state, {**point.timings, **point.commands})
        for i, name in enumerate(self.transient_system.state_names()):
            if name in self._transient_state_indexes:
                dx[self._transient_state_indexes[name]] = transient_derivatives[i]
        self._component_derivatives(dx, point)
        self._controller_derivatives(dx, point)
        self._apply_state_modes(dx, x)
        return dx

    def integrate(self, sample_times: np.ndarray | None = None) -> DAEExecutionResult:
        warm_start = WarmStart(self.Z0.copy())
        time_points = np.asarray(sample_times, dtype=float) if sample_times is not None else self._default_times()
        if time_points.size == 0:
            time_points = np.array([self.execution_plan.time_start_s], dtype=float)
        if self.X0.size == 0:
            X = np.zeros((time_points.size, 0), dtype=float)
        elif time_points.size == 1 or self.execution_plan.time_end_s <= self.execution_plan.time_start_s:
            X = self.X0.reshape(1, -1)
        else:
            sol = solve_ivp(
                lambda t, x: self.rhs(t, x, warm_start),
                (self.execution_plan.time_start_s, self.execution_plan.time_end_s),
                self.X0,
                t_eval=time_points,
                method=self.execution_plan.integration.method,
                rtol=self.execution_plan.integration.rtol,
                atol=self.execution_plan.integration.atol,
                max_step=self.execution_plan.integration.max_step or np.inf,
            )
            if not sol.success:
                raise RuntimeError(f"DAE integration failed: {sol.message}")
            X = sol.y.T
        points = [self.evaluate(float(t), X[i], warm_start) for i, t in enumerate(time_points)]
        return self._result(time_points, X, points)

    def _state_dict(self, x: np.ndarray) -> dict[str, float]:
        values = {name: float(x[i]) for i, name in enumerate(self.state_names)}
        for name, mode in self.execution_plan.state_modes.items():
            if mode.mode == "fixed" and mode.value is not None:
                values[name] = float(mode.value)
        return values

    def _transient_state_vector(self, x: np.ndarray) -> np.ndarray:
        values = []
        for name in self.transient_system.state_names():
            index = self._transient_state_indexes.get(name)
            values.append(float(x[index]) if index is not None else 0.0)
        return np.asarray(values, dtype=float)

    def _network_inputs(
        self,
        states: Mapping[str, float],
        targets: Mapping[str, Any],
        boundaries: Mapping[str, Any],
        timings: Mapping[str, Any],
        commands: Mapping[str, Any],
        transient_sources: Mapping[str, float],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        values.update(states)
        values.update({key: value for key, value in boundaries.items()})
        values.update({f"boundaries.{key}": value for key, value in boundaries.items()})
        values.update({key: value for key, value in timings.items()})
        values.update({f"timings.{key}": value for key, value in timings.items()})
        values.update({key: value for key, value in commands.items()})
        values.update(transient_sources)
        values.update({f"targets.{key}": value for key, value in targets.items()})
        values.update({f"target.{key}": value for key, value in targets.items()})
        return values

    def _evaluate_controllers(
        self,
        t: float,
        targets: Mapping[str, Any],
        timings: Mapping[str, Any],
        measurements: Mapping[str, Any],
        states: Mapping[str, float],
    ) -> dict[str, Any]:
        if self.loaded.controllers is None:
            return {}
        sample_index = controller_sample_index(t, self._controller_period_s)
        if measurements and sample_index is not None and sample_index in self._controller_hold_cache:
            return dict(self._controller_hold_cache[sample_index])
        outputs: dict[str, Any] = {}
        current_phase = self._current_phase(t)
        for name in controller_execution_order(self.loaded.controllers.controllers):
            controller = self.loaded.controllers.controllers[name]
            if not controller_is_active(controller, current_phase):
                continue
            ctype = str(controller.get("type", "null"))
            if ctype in {"proportional", "pi", "pid"}:
                outputs.update(self._feedback_controller(name, controller, targets, timings, measurements, outputs, states, sample_index=sample_index))
            elif ctype == "rate_limiter":
                outputs.update(self._rate_limiter(name, controller, targets, timings, measurements, outputs, states))
            else:
                from atha.config.controllers import evaluate_dynamic_controllers

                single = type("_SingleControllerConfig", (), {"controllers": {name: controller}, "path": self.loaded.controllers.path})()
                outputs.update(
                    evaluate_dynamic_controllers(
                        single,
                        targets,
                        {**timings, **outputs},
                        measurements,
                        dt=self._controller_period_s or 1.0,
                        current_phase=current_phase,
                        previous_outputs=self._controller_hold_cache.get((sample_index or 0) - 1, {}) if sample_index is not None else {},
                    )
                )
        if measurements and sample_index is not None:
            self._controller_hold_cache[sample_index] = dict(outputs)
        return outputs

    def _feedback_controller(
        self,
        name: str,
        controller: Mapping[str, Any],
        targets: Mapping[str, Any],
        timings: Mapping[str, Any],
        measurements: Mapping[str, Any],
        outputs: Mapping[str, Any],
        states: Mapping[str, float],
        sample_index: int | None = None,
    ) -> dict[str, Any]:
        ctype = str(controller.get("type", "proportional"))
        inputs = controller["inputs"]
        params = controller.get("parameters", {})
        target = float(_lookup_signal(str(inputs["target"]), targets, timings, measurements, outputs))
        measurement = float(_lookup_signal(str(inputs["measurement"]), targets, timings, measurements, outputs))
        error = target - measurement
        integral = states.get(f"controller.{name}.integral", float(params.get("integral_initial", 0.0)))
        if ctype == "pid" and self._controller_period_s:
            prev_idx = (sample_index or 0) - 1
            previous_cache = self._controller_hold_cache.get(prev_idx, {})
            previous_error = float(previous_cache.get(f"controller.{name}.error", error))
            derivative = (error - previous_error) / max(self._controller_period_s, 1.0e-12)
        else:
            derivative = 0.0
        proportional_term = float(params.get("gain", params.get("proportional_gain", params.get("kp", 0.0)))) * error
        integral_term = (
            float(params.get("ki", params.get("integral_gain", 0.0))) * integral
            if ctype in {"pi", "pid"}
            else 0.0
        )
        derivative_term = (
            float(params.get("kd", params.get("derivative_gain", 0.0))) * derivative
            if ctype == "pid"
            else 0.0
        )
        raw = (
            float(params.get("bias", 0.0))
            + float(params.get("feed_forward_gain", 0.0)) * target
            + proportional_term
            + integral_term
            + derivative_term
        )
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

    def _rate_limiter(
        self,
        name: str,
        controller: Mapping[str, Any],
        targets: Mapping[str, Any],
        timings: Mapping[str, Any],
        measurements: Mapping[str, Any],
        outputs: Mapping[str, Any],
        states: Mapping[str, float],
    ) -> dict[str, Any]:
        value = float(_lookup_signal(str(controller["input"]), targets, timings, measurements, outputs))
        previous = float(states.get(f"controller.{name}.previous_command", controller.get("parameters", {}).get("initial", value)))
        return {
            str(controller["output"]): previous,
            f"controller.{name}.command": previous,
            f"controller.{name}.target_command": value,
            f"controller.{name}.rate": value - previous,
        }

    def _controller_derivatives(self, dx: np.ndarray, point: DAEPoint) -> None:
        if self.loaded.controllers is None:
            return
        for name, controller in self.loaded.controllers.controllers.items():
            ctype = str(controller.get("type", "null"))
            if ctype in {"pi", "pid"}:
                state_name = f"controller.{name}.integral"
                index = self._controller_state_indexes.get(state_name)
                if index is not None:
                    saturated = point.commands.get(f"controller.{name}.saturated", 0.0)
                    anti_windup = bool(controller.get("parameters", {}).get("anti_windup", True))
                    dx[index] = 0.0 if anti_windup and saturated else point.commands.get(f"controller.{name}.error", 0.0)
            if ctype == "rate_limiter":
                state_name = f"controller.{name}.previous_command"
                index = self._controller_state_indexes.get(state_name)
                if index is not None:
                    params = controller.get("parameters", {})
                    target = point.commands.get(f"controller.{name}.target_command", point.states.get(state_name, 0.0))
                    error = target - point.states.get(state_name, 0.0)
                    opening = float(params.get("opening_rate", params.get("rate", 1.0)))
                    closing = float(params.get("closing_rate", params.get("rate", opening)))
                    rate = opening if error >= 0.0 else closing
                    dx[index] = np.sign(error) * min(abs(error), abs(rate))

    def _component_derivatives(self, dx: np.ndarray, point: DAEPoint) -> None:
        values = {**point.states, **point.algebraics, **point.measurements}
        for component in self.loaded.engine.components.values():
            if component.type == "Pipe":
                self._pipe_derivative(dx, component.name, component.parameters, point.states, values)
            elif component.type in {"CombustionChamber", "Preburner"}:
                self._finite_volume_derivative(dx, component.name, component.parameters, values)
            elif component.type == "Rotor":
                self._rotor_derivative(dx, component.name, component.parameters, values)

    def _pipe_derivative(
        self,
        dx: np.ndarray,
        name: str,
        params: Mapping[str, Any],
        states: Mapping[str, float],
        values: Mapping[str, float],
    ) -> None:
        index = self._state_index(f"{name}.mdot")
        if index is None:
            return
        tau = max(float(params.get("time_constant", params.get("tau", 0.0))), 0.0)
        target = float(values.get(f"{name}.mdot", values.get(f"{name}.mdot_steady", values.get(f"{name}.inlet.mdot", 0.0))))
        current = float(states.get(f"{name}.mdot", self.X0[index] if index < self.X0.size else 0.0))
        dx[index] = 0.0 if tau <= 0.0 else (target - current) / tau

    def _finite_volume_derivative(
        self,
        dx: np.ndarray,
        name: str,
        params: Mapping[str, Any],
        values: Mapping[str, float],
    ) -> None:
        index = self._state_index(f"{name}.P")
        if index is None:
            return
        volume = max(float(params.get("volume", 0.0)), 1.0e-12)
        gas_r = float(params.get("gas_R", 287.0))
        temperature = float(values.get(f"{name}.T", params.get("T_adiabatic", params.get("initial_T", 300.0))))
        mdot_in = _sum_component_ports(values, name, ("fuel_inlet", "ox_inlet", "lox_inlet", "inlet"))
        mdot_out = _sum_component_ports(values, name, ("outlet",))
        dx[index] = gas_r * temperature / volume * (mdot_in - mdot_out)

    def _rotor_derivative(
        self,
        dx: np.ndarray,
        name: str,
        params: Mapping[str, Any],
        values: Mapping[str, float],
    ) -> None:
        index = self._state_index(f"{name}.omega")
        if index is None:
            return
        inertia = max(float(params.get("moment_of_inertia", params.get("inertia", 1.0))), 1.0e-12)
        omega = float(values.get(f"{name}.omega", values.get(f"{name}.shaft.omega", 0.0)))
        omega_abs = max(abs(omega), 1.0)
        drive_power = sum(_turbine_power(component, values) for component in self._shaft_couplings.get(name, {}).get("turbines", ()))
        load_power = sum(_pump_power(self.loaded.engine.components[component], values) for component in self._shaft_couplings.get(name, {}).get("pumps", ()))
        friction = float(params.get("friction_coeff", 0.0))
        dx[index] = ((drive_power - load_power) / omega_abs - friction * omega) / inertia

    def _state_index(self, name: str) -> int | None:
        try:
            return self.state_names.index(name)
        except ValueError:
            return None

    def _current_phase(self, t: float) -> str | None:
        for phase in self.execution_plan.phases:
            name = getattr(phase, "name", "")
            if not name:
                continue
            start = float(getattr(phase, "start_s"))
            end = float(getattr(phase, "end_s"))
            if start <= float(t) < end or (float(t) == end and end == self.execution_plan.time_end_s):
                return str(name)
        return None

    def _apply_state_modes(self, dx: np.ndarray, x: np.ndarray) -> None:
        for name, mode in self.execution_plan.state_modes.items():
            if name not in self.state_names:
                continue
            index = self.state_names.index(name)
            if mode.mode in {"inactive", "fixed", "steady_state"}:
                dx[index] = 0.0

    def _measurements(self, algebraics: Mapping[str, float], states: Mapping[str, float]) -> dict[str, float]:
        measurements = {**states, **algebraics}
        measurements.setdefault("mdot_total", float(algebraics.get("mdot.total", algebraics.get("nozzle.mdot", 0.0))))
        if "chamber.OF" in algebraics:
            measurements.setdefault("OF", float(algebraics["chamber.OF"]))
        for name, component in self.loaded.engine.components.items():
            if component.type == "Rotor":
                omega = measurements.get(f"{name}.omega")
                if omega is not None:
                    measurements.setdefault(f"{name}.rpm", float(omega) * 60.0 / (2.0 * np.pi))
            elif component.type == "Pump":
                power = _pump_power(component, measurements)
                if power:
                    measurements.setdefault(f"{name}.power", power)
                    omega = measurements.get(f"{name}.shaft.omega", measurements.get(f"{name}.omega", 0.0))
                    measurements.setdefault(f"{name}.tau_load", power / max(abs(float(omega)), 1.0))
            elif component.type == "Turbine":
                power = _turbine_power(name, measurements)
                if power:
                    measurements.setdefault(f"{name}.tau_drive", power / max(abs(float(measurements.get(f"{name}.shaft.omega", 0.0))), 1.0))
            elif component.type == "Nozzle":
                if f"{name}.thrust" not in measurements:
                    p_in = measurements.get(f"{name}.inlet.P", measurements.get("chamber.P", 0.0))
                    p_amb = measurements.get(f"{name}.ambient.P", measurements.get("nozzle.ambient.P", 101325.0))
                    cf = float(component.parameters.get("thrust_coefficient", 1.0))
                    area = float(component.parameters.get("throat_area", 0.0))
                    measurements[f"{name}.thrust"] = cf * area * max(float(p_in) - float(p_amb), 0.0)
        return {key: float(value) for key, value in measurements.items() if _is_number(value)}

    def _default_times(self) -> np.ndarray:
        start = self.execution_plan.time_start_s
        end = self.execution_plan.time_end_s
        if end <= start:
            return np.array([start], dtype=float)
        return np.linspace(start, end, 101)

    def _result(self, time: np.ndarray, X: np.ndarray, points: list[DAEPoint]) -> DAEExecutionResult:
        Z = np.vstack([[point.algebraics.get(name, np.nan) for name in self.algebraic_names] for point in points])
        return DAEExecutionResult(
            time=time,
            state_names=self.state_names,
            algebraic_names=self.algebraic_names,
            residual_names=self.network_problem.residual_names,
            X=X,
            Z=Z,
            residual_history=_history(points, "normalized_residuals"),
            command_history=_history(points, "commands"),
            target_history=_history(points, "targets"),
            boundary_history=_history(points, "boundaries"),
            measurement_history=_history(points, "measurements"),
            points=points,
        )


def _history(points: list[DAEPoint], attr: str) -> dict[str, np.ndarray]:
    keys: set[str] = set()
    for point in points:
        values = getattr(point, attr)
        keys.update(key for key, value in values.items() if _is_number(value))
    return {
        key: np.asarray([float(getattr(point, attr).get(key, np.nan)) for point in points], dtype=float)
        for key in sorted(keys)
    }


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
        return measurements.get(path[len("measurements."):], 0.0)
    if path in commands:
        return commands[path]
    if path in targets:
        return targets[path]
    if path in timings:
        return timings[path]
    if path in measurements:
        return measurements[path]
    raise ValueError(f"Unknown controller signal: {path}")


def _lookup_nested(data: Mapping[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        value = value[part]
    return value


def _sum_component_ports(values: Mapping[str, float], component: str, ports: tuple[str, ...]) -> float:
    return sum(float(values.get(f"{component}.{port}.mdot", 0.0)) for port in ports)


def _shaft_couplings(components: Mapping[str, Any], connections: list[Any]) -> dict[str, dict[str, list[str]]]:
    couplings: dict[str, dict[str, list[str]]] = {}
    for connection in connections:
        if getattr(connection, "domain", "") != "shaft":
            continue
        source_component, _source_port = str(connection.source).split(".", 1)
        target_component, _target_port = str(connection.target).split(".", 1)
        source_type = components[source_component].type if source_component in components else ""
        target_type = components[target_component].type if target_component in components else ""
        rotor = source_component if source_type == "Rotor" else target_component if target_type == "Rotor" else None
        other = target_component if rotor == source_component else source_component if rotor == target_component else None
        if rotor is None or other is None or other not in components:
            continue
        bucket = couplings.setdefault(rotor, {"pumps": [], "turbines": []})
        other_type = components[other].type
        if other_type == "Pump":
            bucket["pumps"].append(other)
        elif other_type == "Turbine":
            bucket["turbines"].append(other)
    return couplings


def _pump_power(component: Any, values: Mapping[str, float]) -> float:
    name = component.name
    if f"{name}.power" in values:
        return float(values[f"{name}.power"])
    mdot = _first_numeric(
        values,
        (
            f"{name}.mdot",
            f"{name}.inlet.mdot",
            f"{name}.outlet.mdot",
            f"{name}.inlet_mdot",
        ),
        float(component.parameters.get("pump_map", {}).get("mdot_design", 0.0)) if isinstance(component.parameters.get("pump_map", {}), Mapping) else 0.0,
    )
    delta_p = _first_numeric(
        values,
        (f"{name}.delta_P", f"{name}.pressure_rise"),
        float(component.parameters.get("pump_map", {}).get("dP_design", component.parameters.get("delta_P_design", 0.0)))
        if isinstance(component.parameters.get("pump_map", {}), Mapping)
        else float(component.parameters.get("delta_P_design", 0.0)),
    )
    eta = _pump_efficiency(component, values)
    return abs(mdot) * max(delta_p, 0.0) / max(eta, 1.0e-6)


def _pump_efficiency(component: Any, values: Mapping[str, float]) -> float:
    name = component.name
    if f"{name}.efficiency" in values:
        return min(max(float(values[f"{name}.efficiency"]), 1.0e-6), 1.0)
    pump_map = component.parameters.get("pump_map", {})
    if isinstance(pump_map, Mapping):
        return min(max(float(pump_map.get("efficiency_design", component.parameters.get("efficiency_design", 0.7))), 1.0e-6), 1.0)
    return min(max(float(component.parameters.get("efficiency_design", 0.7)), 1.0e-6), 1.0)


def _turbine_power(component: str, values: Mapping[str, float]) -> float:
    return max(float(values.get(f"{component}.power", 0.0)), 0.0)


def _first_numeric(values: Mapping[str, float], paths: tuple[str, ...], default: float) -> float:
    for path in paths:
        if path in values and _is_number(values[path]):
            return float(values[path])
    return float(default)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating))
