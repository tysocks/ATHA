from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from atha.assembly import EngineAssembler
from atha.components.derivatives import DerivativeEvaluationContext
from atha.components.registry import component_derivative_contract, component_spec
from atha.config import (
    TransientSystem,
    balance_configs,
    controller_execution_order,
    evaluate_boundary_conditions,
    evaluate_operating_targets,
    evaluate_timing_events,
)
from atha.config.schedules import collect_config_breakpoints
from atha.config.controllers import controller_reset_state_values, controller_state_infos
from atha.config.controllers import controller_evaluation_period, controller_is_active, controller_sample_index
from atha.config.mission_phases import (
    controller_hold_when_inactive,
    controller_should_reset_on_enter,
    detect_phase_transition,
    resolve_phase_name,
)
from atha.config.loader import LoadedAnalysisConfig
from atha.network import NetworkProblem, NetworkSolution, WarmStart
from atha.network.preconditioner import precondition_algebraic_guess
from atha.runner.progress import SolverProgressEvent
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
    segments: list["IntegrationSegment"] = field(default_factory=list)


@dataclass(frozen=True)
class IntegrationSegment:
    start_s: float
    end_s: float
    reason: str = "scheduled_breakpoint"


@dataclass(frozen=True)
class DAESolvePolicy:
    allow_non_square: bool = False
    checked: bool = True
    residual_tolerance: float = 1.0e-8
    max_nfev: int | None = None
    strict_sources: bool = False
    corrector: str = "none"
    corrector_iterations: int = 1
    preconditioner: str = "none"
    algebraic_solver: str = "nonlinear"


class DAEExecutionProblem:
    """Universal DAE execution-loop foundation.

    This class owns the common `X/Z/Rz` bookkeeping for the production ATHA
    path. It evaluates schedules, mission phases, controllers, transient blocks,
    state modes, and the port algebraic network through one path.

    Plant derivatives are supplied by registered component derivative contracts
    (Pipe, combustors/GG/preburner, Rotor, RegenChannel, GasVolume). Transient
    actuator states and controller memory are also integrated. Components without
    a derivative contract remain algebraically closed (or static) by design.
    """

    def __init__(
        self,
        loaded: LoadedAnalysisConfig,
        execution_plan: ExecutionPlan,
        *,
        network_problem: NetworkProblem | None = None,
        progress_callback: Callable[[SolverProgressEvent], None] | None = None,
    ) -> None:
        self.loaded = loaded
        self.execution_plan = execution_plan
        self.assembler = EngineAssembler(loaded)
        self.solve_policy = _dae_solve_policy(loaded, execution_plan)
        self.progress_callback = progress_callback
        self._last_progress_percent = -1.0e9
        self.solver_source = "generic_port"
        self.network_problem = network_problem or self.assembler.port_network_problem(
            require_square=not self.solve_policy.allow_non_square
        )
        initial_vectors = self.assembler.initial_vectors()
        self.state_names = initial_vectors.state_names
        self.X0 = np.asarray(initial_vectors.X, dtype=float)
        self.algebraic_names = self.network_problem.variable_names
        self.Z0 = self.network_problem.initial_z.copy()
        self._apply_initial_algebraic_overrides()
        self.Z0 = self._precondition_guess(0.0, self.Z0, {})
        if self.solve_policy.strict_sources:
            _validate_strict_full_port_sources(loaded)
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
        self._inactive_command_hold: dict[str, float] = {}
        self._previous_phase: str | None = None
        self._shaft_couplings = _shaft_couplings(loaded.engine.components, loaded.engine.connections)
        self.balances = balance_configs(loaded.analysis_config.analysis.get("balances", {}))
        self._initial_trimmed = False

    def initial_state(self) -> np.ndarray:
        return self.X0.copy()

    def trim_initial_conditions(self, t: float | None = None) -> DAEPoint:
        trim_time = self.execution_plan.time_start_s if t is None else float(t)
        point = self.evaluate(trim_time, self.X0.copy(), self.Z0.copy())
        self.Z0 = np.asarray(
            [point.algebraics.get(name, self.Z0[i]) for i, name in enumerate(self.algebraic_names)],
            dtype=float,
        )
        for i, name in enumerate(self.state_names):
            if name in point.algebraics:
                self.X0[i] = float(point.algebraics[name])
        self._initial_trimmed = True
        return point

    def evaluate(self, t: float, x: np.ndarray, warm_start: WarmStart | np.ndarray | None = None) -> DAEPoint:
        states = self._state_dict(x)
        targets = evaluate_operating_targets(self.loaded.operating_conditions, t) if self.loaded.operating_conditions is not None else {}
        boundaries = evaluate_boundary_conditions(self.loaded.boundary_conditions, t) if self.loaded.boundary_conditions is not None else {}
        timings = evaluate_timing_events(self.loaded.timings, t)
        transient_state = self._transient_state_vector(x)
        commands = self._evaluate_controllers(t, targets, timings, {}, states)
        transient_sources = self.transient_system.sample_sources(t, transient_state, {**timings, **commands})
        network_inputs = self._network_inputs(t, states, targets, boundaries, timings, commands, transient_sources)
        solution = self._solve_network(t, warm_start, network_inputs)
        measurements = self._measurements(solution.values, states)
        commands = self._evaluate_controllers(t, targets, timings, measurements, states)
        transient_sources = self.transient_system.sample_sources(t, transient_state, {**timings, **commands})
        network_inputs = self._network_inputs(t, states, targets, boundaries, timings, commands, transient_sources)
        solution = self._solve_network(t, warm_start, network_inputs)
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
        # Apply phase-entry controller memory resets only on the integration path.
        self._apply_phase_entry_resets_to_state(t, x)
        point = self.evaluate(t, x, warm_start)
        name, value = _largest_abs(point.normalized_residuals)
        self._emit_progress(
            "progress",
            t,
            "integrating",
            residual_name=name,
            residual_value=value,
        )
        dx = self._raw_rhs_from_point(t, x, point)
        self._apply_state_modes(dx, x)
        return dx

    def _raw_rhs_from_point(self, t: float, x: np.ndarray, point: DAEPoint) -> np.ndarray:
        dx = np.zeros_like(x, dtype=float)
        transient_state = self._transient_state_vector(x)
        transient_derivatives = self.transient_system.derivatives(t, transient_state, {**point.timings, **point.commands})
        for i, name in enumerate(self.transient_system.state_names()):
            if name in self._transient_state_indexes:
                dx[self._transient_state_indexes[name]] = transient_derivatives[i]
        self._component_derivatives(dx, point)
        self._controller_derivatives(dx, point)
        return dx

    def integrate(self, sample_times: np.ndarray | None = None) -> DAEExecutionResult:
        if not self._initial_trimmed:
            self.trim_initial_conditions()
        self._emit_progress("setup", self.execution_plan.time_start_s, "building sample schedule", force=True)
        time_points = np.asarray(sample_times, dtype=float) if sample_times is not None else self._default_times()
        if time_points.size == 0:
            time_points = np.array([self.execution_plan.time_start_s], dtype=float)
        time_points = _unique_sorted_clipped(time_points, self.execution_plan.time_start_s, self.execution_plan.time_end_s)
        segments = self.integration_segments(
            time_points if self.execution_plan.integration.segment_at_samples else None
        )
        self._emit_progress(
            "setup",
            self.execution_plan.time_start_s,
            f"{len(segments)} integration segment(s), {len(time_points)} output sample(s)",
            force=True,
        )
        X = self._integrate_segments(time_points, segments)
        self._emit_progress("progress", self.execution_plan.time_end_s, "evaluating output samples", force=True)
        points = self._evaluate_points(time_points, X)
        self._emit_progress("complete", self.execution_plan.time_end_s, "integration complete", force=True)
        return self._result(time_points, X, points, segments)

    def integration_segments(self, sample_times: np.ndarray | None = None) -> list[IntegrationSegment]:
        breakpoints = self._integration_breakpoints(sample_times)
        return [
            IntegrationSegment(
                float(breakpoints[i]),
                float(breakpoints[i + 1]),
                reason=self._segment_reason(float(breakpoints[i]), float(breakpoints[i + 1])),
            )
            for i in range(len(breakpoints) - 1)
            if breakpoints[i + 1] > breakpoints[i]
        ] or [IntegrationSegment(self.execution_plan.time_start_s, self.execution_plan.time_end_s)]

    def _segment_reason(self, start: float, end: float) -> str:
        for phase in self.execution_plan.phases:
            phase_start = float(phase.start_s)
            phase_end = float(phase.end_s)
            if (
                str(phase.name)
                and np.isclose(start, phase_start, rtol=0.0, atol=1.0e-12)
                and np.isclose(end, phase_end, rtol=0.0, atol=1.0e-12)
            ):
                return f"phase:{phase.name}"
        for phase in self.execution_plan.phases:
            phase_start = float(phase.start_s)
            phase_end = float(phase.end_s)
            if str(phase.name) and start >= phase_start - 1.0e-12 and end <= phase_end + 1.0e-12:
                return f"phase:{phase.name}:scheduled_breakpoint"
        return "scheduled_breakpoint"

    def _integration_breakpoints(self, sample_times: np.ndarray | None = None) -> list[float]:
        start = float(self.execution_plan.time_start_s)
        end = float(self.execution_plan.time_end_s)
        points = {start, end}
        for phase in self.execution_plan.phases:
            points.add(float(phase.start_s))
            points.add(float(phase.end_s))
        for point in collect_config_breakpoints(
            self.loaded.boundary_conditions,
            self.loaded.timings,
            self.loaded.operating_conditions,
            t_start=start,
            t_end=end,
        ):
            points.add(float(point))
        if self._controller_period_s is not None and self.execution_plan.integration.segment_at_controller_samples:
            k0 = int(np.ceil(start / self._controller_period_s))
            k1 = int(np.floor(end / self._controller_period_s))
            for k in range(k0, k1 + 1):
                points.add(float(k * self._controller_period_s))
        if sample_times is not None:
            for point in sample_times:
                if start <= float(point) <= end:
                    points.add(float(point))
        return sorted(point for point in points if start <= point <= end)

    def _integrate_segments(self, time_points: np.ndarray, segments: list[IntegrationSegment]) -> np.ndarray:
        if self.X0.size == 0:
            return np.zeros((time_points.size, 0), dtype=float)
        if time_points.size == 1 or self.execution_plan.time_end_s <= self.execution_plan.time_start_s:
            return self.X0.reshape(1, -1)

        rows: dict[float, np.ndarray] = {}
        current_x = self.X0.copy()
        current_z = self.Z0.copy()
        if np.any(np.isclose(time_points, self.execution_plan.time_start_s, rtol=0.0, atol=1.0e-12)):
            rows[float(self.execution_plan.time_start_s)] = current_x.copy()

        for segment in segments:
            segment_start = float(segment.start_s)
            segment_end = float(segment.end_s)
            self._emit_progress("segment", segment_start, f"starting {segment.reason}", force=True)
            segment_times = [
                float(t)
                for t in time_points
                if segment_start < float(t) <= segment_end + 1.0e-12
            ]
            eval_times = _unique_sorted_clipped(
                np.asarray([*segment_times, segment_end], dtype=float),
                segment_start,
                segment_end,
            )
            if segment_end <= segment_start:
                continue
            if eval_times.size == 0:
                eval_times = np.asarray([segment_end], dtype=float)
            z_guess = current_z.copy()
            if self.execution_plan.integration.method.lower() in {"fixed_rk4", "rk4_fixed"}:
                fixed = self._integrate_fixed_rk4(segment_start, segment_end, current_x, eval_times, z_guess)
                for t_value, x_value in fixed.items():
                    if any(np.isclose(t_value, requested, rtol=0.0, atol=1.0e-12) for requested in segment_times):
                        rows[float(t_value)] = x_value.copy()
                current_x = fixed[float(eval_times[-1])].copy()
            else:
                sol = solve_ivp(
                    lambda t, x: self.rhs(t, x, z_guess.copy()),
                    (segment_start, segment_end),
                    current_x,
                    t_eval=eval_times,
                    method=self.execution_plan.integration.method,
                    rtol=self.execution_plan.integration.rtol,
                    atol=self.execution_plan.integration.atol,
                    max_step=self.execution_plan.integration.max_step or np.inf,
                )
                if not sol.success:
                    raise RuntimeError(f"DAE integration failed in segment {segment_start:g}-{segment_end:g}: {sol.message}")
                for index, t_value in enumerate(sol.t):
                    if any(np.isclose(t_value, requested, rtol=0.0, atol=1.0e-12) for requested in segment_times):
                        rows[float(t_value)] = sol.y[:, index].copy()
                current_x = sol.y[:, -1].copy()
            current_x, current_z, end_point = self._correct_segment_endpoint(segment_start, segment_end, current_x, current_z)
            current_z = np.asarray([end_point.algebraics.get(name, current_z[i]) for i, name in enumerate(self.algebraic_names)], dtype=float)
            largest_name, largest_value = _largest_abs(end_point.normalized_residuals)
            self._emit_progress(
                "progress",
                segment_end,
                f"completed {segment.reason}",
                residual_name=largest_name,
                residual_value=largest_value,
                force=True,
            )

        return np.vstack([rows.get(float(t), current_x.copy()) for t in time_points])

    def _integrate_fixed_rk4(
        self,
        start: float,
        end: float,
        x0: np.ndarray,
        eval_times: np.ndarray,
        z_guess: np.ndarray,
    ) -> dict[float, np.ndarray]:
        max_step = self.execution_plan.integration.max_step or max(end - start, 1.0e-6)
        output: dict[float, np.ndarray] = {}
        x = np.asarray(x0, dtype=float).copy()
        current = float(start)
        for target in eval_times:
            target = float(target)
            while current < target - 1.0e-12:
                step = min(float(max_step), target - current)
                k1 = self.rhs(current, x, z_guess.copy())
                k2 = self.rhs(current + 0.5 * step, self._project_state_bounds(x + 0.5 * step * k1), z_guess.copy())
                k3 = self.rhs(current + 0.5 * step, self._project_state_bounds(x + 0.5 * step * k2), z_guess.copy())
                k4 = self.rhs(current + step, self._project_state_bounds(x + step * k3), z_guess.copy())
                x = self._project_state_bounds(x + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
                current += step
            output[target] = x.copy()
        return output

    def _project_state_bounds(self, x: np.ndarray) -> np.ndarray:
        bounded = np.asarray(x, dtype=float).copy()
        components = self.loaded.engine.components
        transient_bounds = self._transient_state_bounds()
        for i, name in enumerate(self.state_names):
            if name.endswith(".P"):
                component_name = name.split(".", 1)[0]
                component = components.get(component_name)
                floor = 1.0
                if component is not None:
                    floor = float(component.parameters.get("pressure_floor", component.parameters.get("min_pressure", floor)))
                if bounded[i] < floor:
                    bounded[i] = floor
            elif name.endswith(".omega"):
                component_name = name.split(".", 1)[0]
                component = components.get(component_name)
                if component is not None and component.type == "Rotor" and not bool(component.parameters.get("allow_reverse", False)):
                    floor = float(component.parameters.get("min_omega", 0.0))
                    if bounded[i] < floor:
                        bounded[i] = floor
            if name in transient_bounds:
                lower, upper = transient_bounds[name]
                if lower is not None and bounded[i] < lower:
                    bounded[i] = lower
                if upper is not None and bounded[i] > upper:
                    bounded[i] = upper
        return bounded

    def _transient_state_bounds(self) -> dict[str, tuple[float | None, float | None]]:
        bounds: dict[str, tuple[float | None, float | None]] = {}
        for block in self.transient_system.blocks:
            names = block.state_names
            if not names:
                continue
            lower = block.config.parameters.get("lower_limit", block.config.state.get("lower_limit"))
            upper = block.config.parameters.get("upper_limit", block.config.state.get("upper_limit"))
            bounds[names[0]] = (
                float(lower) if lower is not None else None,
                float(upper) if upper is not None else None,
            )
        return bounds

    def _evaluate_points(self, time_points: np.ndarray, X: np.ndarray) -> list[DAEPoint]:
        z_guess = self.Z0.copy()
        points: list[DAEPoint] = []
        for i, t in enumerate(time_points):
            point = self.evaluate(float(t), X[i], z_guess.copy())
            z_guess = np.asarray([point.algebraics.get(name, z_guess[j]) for j, name in enumerate(self.algebraic_names)], dtype=float)
            self._add_state_mode_residuals(point, X[i])
            points.append(point)
        return points

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
        t: float,
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
        values["time"] = float(t)
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
        current_phase = self._current_phase(t)
        sample_index = controller_sample_index(t, self._controller_period_s)
        if measurements and sample_index is not None and sample_index in self._controller_hold_cache:
            self._previous_phase = current_phase
            return dict(self._controller_hold_cache[sample_index])
        if not measurements and sample_index is not None:
            if sample_index in self._controller_hold_cache:
                self._previous_phase = current_phase
                return dict(self._controller_hold_cache[sample_index])
            previous = self._controller_hold_cache.get(sample_index - 1)
            if previous is not None:
                self._previous_phase = current_phase
                return dict(previous)
        outputs: dict[str, Any] = {}
        for name in controller_execution_order(self.loaded.controllers.controllers):
            controller = self.loaded.controllers.controllers[name]
            if not controller_is_active(controller, current_phase):
                if controller_hold_when_inactive(controller):
                    outputs.update(self._inactive_controller_hold(name, controller))
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
            self._capture_inactive_hold(name, controller, outputs)
        if measurements and sample_index is not None:
            self._controller_hold_cache[sample_index] = dict(outputs)
        self._previous_phase = current_phase
        return outputs

    def _apply_phase_entry_resets_to_state(self, t: float, x: np.ndarray) -> None:
        if self.loaded.controllers is None:
            return
        current_phase = self._current_phase(t)
        transition = detect_phase_transition(self._previous_phase, current_phase, t)
        if transition is None or not transition.entered or current_phase is None:
            return
        reset_applied = False
        for name, controller in self.loaded.controllers.controllers.items():
            if not controller_should_reset_on_enter(controller, current_phase):
                continue
            for state_name, value in controller_reset_state_values(name, controller).items():
                index = self._controller_state_indexes.get(state_name)
                if index is not None:
                    x[index] = value
                    self.X0[index] = value
                    reset_applied = True
        if reset_applied:
            self._controller_hold_cache.clear()

    def _capture_inactive_hold(self, name: str, controller: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
        output = controller.get("output")
        if isinstance(output, str) and output in outputs and isinstance(outputs[output], (int, float, np.floating)):
            self._inactive_command_hold[output] = float(outputs[output])
        output_map = controller.get("outputs")
        if isinstance(output_map, Mapping):
            for path in output_map.values():
                key = str(path)
                if key in outputs and isinstance(outputs[key], (int, float, np.floating)):
                    self._inactive_command_hold[key] = float(outputs[key])

    def _inactive_controller_hold(self, name: str, controller: Mapping[str, Any]) -> dict[str, float]:
        held: dict[str, float] = {}
        output = controller.get("output")
        if isinstance(output, str) and output in self._inactive_command_hold:
            held[output] = self._inactive_command_hold[output]
            held[f"controller.{name}.command"] = held[output]
            held[f"controller.{name}.held"] = 1.0
        output_map = controller.get("outputs")
        if isinstance(output_map, Mapping):
            for path in output_map.values():
                key = str(path)
                if key in self._inactive_command_hold:
                    held[key] = self._inactive_command_hold[key]
        return held

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
        context = DerivativeEvaluationContext(
            states=point.states,
            algebraics=point.algebraics,
            measurements=point.measurements,
            inputs={
                **point.commands,
                **point.targets,
                **point.boundaries,
                **point.timings,
            },
            shaft_couplings={
                key: {bucket: tuple(values) for bucket, values in coupling.items()}
                for key, coupling in self._shaft_couplings.items()
            },
        )
        for component in self.loaded.engine.components.values():
            contract = component_derivative_contract(component)
            if contract is None:
                continue
            for path, derivative in contract.derivatives(component, context).items():
                index = self._state_index(path)
                if index is not None:
                    dx[index] = float(derivative)

    def _state_index(self, name: str) -> int | None:
        try:
            return self.state_names.index(name)
        except ValueError:
            return None

    def _apply_initial_state_overrides(self, overrides: Mapping[str, float]) -> None:
        for name, value in overrides.items():
            if name in self.state_names:
                self.X0[self.state_names.index(name)] = float(value)
            else:
                self.state_names.append(str(name))
                self.X0 = np.concatenate((self.X0, np.asarray([float(value)], dtype=float)))

    def _apply_initial_algebraic_overrides(self) -> None:
        overrides = self.loaded.analysis_config.analysis.get("initial_algebraic", {})
        if not isinstance(overrides, Mapping):
            return
        for name, value in overrides.items():
            if name in self.algebraic_names and _is_number(value):
                self.Z0[self.algebraic_names.index(str(name))] = float(value)

    def _correct_segment_endpoint(
        self,
        start: float,
        end: float,
        x: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, DAEPoint]:
        point = self.evaluate(end, x, z.copy())
        if self.solve_policy.corrector not in {"endpoint", "backward_euler"} or end <= start:
            return x, np.asarray([point.algebraics.get(name, z[i]) for i, name in enumerate(self.algebraic_names)], dtype=float), point
        dt = float(end - start)
        corrected_x = np.asarray(x, dtype=float).copy()
        corrected_z = np.asarray([point.algebraics.get(name, z[i]) for i, name in enumerate(self.algebraic_names)], dtype=float)
        for _ in range(max(self.solve_policy.corrector_iterations, 1)):
            point = self.evaluate(end, corrected_x, corrected_z.copy())
            rhs = self._raw_rhs_from_point(end, corrected_x, point)
            self._apply_state_modes(rhs, corrected_x)
            candidate_x = np.asarray(x, dtype=float) + dt * rhs
            for index in (*self._transient_state_indexes.values(), *self._controller_state_indexes.values()):
                candidate_x[index] = x[index]
            corrected_x = self._project_state_bounds(candidate_x)
            corrected_z = np.asarray([point.algebraics.get(name, corrected_z[i]) for i, name in enumerate(self.algebraic_names)], dtype=float)
        point = self.evaluate(end, corrected_x, corrected_z.copy())
        corrected_z = np.asarray([point.algebraics.get(name, corrected_z[i]) for i, name in enumerate(self.algebraic_names)], dtype=float)
        return corrected_x, corrected_z, point

    def _current_phase(self, t: float) -> str | None:
        return resolve_phase_name(self.execution_plan.phases, t, self.execution_plan.time_end_s)

    def _apply_state_modes(self, dx: np.ndarray, x: np.ndarray) -> None:
        for name, mode in self.execution_plan.state_modes.items():
            if name not in self.state_names:
                continue
            index = self.state_names.index(name)
            if mode.mode in {"inactive", "fixed", "steady_state"}:
                dx[index] = 0.0

    def _emit_progress(
        self,
        kind: str,
        t: float,
        message: str,
        *,
        residual_name: str | None = None,
        residual_value: float | None = None,
        force: bool = False,
    ) -> None:
        if self.progress_callback is None:
            return
        start = float(self.execution_plan.time_start_s)
        end = float(self.execution_plan.time_end_s)
        percent = 100.0 if end <= start else 100.0 * (float(t) - start) / max(end - start, 1.0e-30)
        percent = min(max(percent, 0.0), 100.0)
        if not force and kind == "progress" and percent < self._last_progress_percent + 0.25:
            return
        if kind == "progress":
            self._last_progress_percent = percent
        self.progress_callback(
            SolverProgressEvent(
                kind=kind,
                message=message,
                time_s=float(t),
                percent=percent,
                phase=self._current_phase(float(t)),
                residual_name=residual_name,
                residual_value=residual_value,
            )
        )

    def _add_state_mode_residuals(self, point: DAEPoint, x: np.ndarray) -> None:
        raw_dx = self._raw_rhs_from_point(point.time, x, point)
        for name, mode in self.execution_plan.state_modes.items():
            if mode.mode != "steady_state" or name not in self.state_names:
                continue
            index = self.state_names.index(name)
            residual_name = f"state_modes.{name}.steady_state_residual"
            point.residuals[residual_name] = float(raw_dx[index])
            point.normalized_residuals[residual_name] = float(raw_dx[index])

    def _measurements(self, algebraics: Mapping[str, float], states: Mapping[str, float]) -> dict[str, float]:
        measurements = {**algebraics, **states}
        mdot_total = float(algebraics.get("mdot.total", algebraics.get("nozzle.mdot", 0.0)))
        measurements.setdefault("mdot_total", mdot_total)
        measurements.setdefault("mdot.total", mdot_total)
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
                omega_value = measurements.get(f"{name}.shaft.omega", measurements.get(f"{name}.omega"))
                if omega_value is not None:
                    measurements.setdefault(f"{name}.rpm", float(omega_value) * 60.0 / (2.0 * np.pi))
                diameter = float(component.parameters.get("diameter", 0.0))
                omega = abs(float(measurements.get(f"{name}.shaft.omega", measurements.get(f"{name}.omega", 0.0))))
                rho = max(float(measurements.get(f"{name}.inlet.rho", 1.0)), 1.0e-12)
                mdot = abs(float(measurements.get(f"{name}.mdot", measurements.get(f"{name}.inlet.mdot", 0.0))))
                if diameter > 0.0 and omega > 0.0:
                    measurements.setdefault(f"{name}.phi", mdot / max(rho * omega * diameter**3, 1.0e-30))
                    measurements.setdefault(
                        f"{name}.psi",
                        float(measurements.get(f"{name}.delta_P", 0.0)) / max(rho * omega**2 * diameter**2, 1.0e-30),
                    )
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
        for key, value in list(measurements.items()):
            if key.endswith(".command") and ("speed" in key or "omega" in key):
                measurements.setdefault(f"{key}.rpm", float(value) * 60.0 / (2.0 * np.pi))
        return {key: float(value) for key, value in measurements.items() if _is_number(value)}

    def _default_times(self) -> np.ndarray:
        start = self.execution_plan.time_start_s
        end = self.execution_plan.time_end_s
        if end <= start:
            return np.array([start], dtype=float)
        return np.linspace(start, end, 101)

    def _result(
        self,
        time: np.ndarray,
        X: np.ndarray,
        points: list[DAEPoint],
        segments: list[IntegrationSegment],
    ) -> DAEExecutionResult:
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
            segments=segments,
        )

    def _solve_network(
        self,
        t: float,
        warm_start: WarmStart | np.ndarray | None,
        network_inputs: Mapping[str, Any],
    ):
        warm_start = self._precondition_warm_start(t, warm_start, network_inputs)
        if self.solve_policy.algebraic_solver in {"preconditioned_direct", "direct", "explicit"}:
            return self._direct_network_solution(t, warm_start, network_inputs)
        if self.solve_policy.checked:
            return self.network_problem.solve_checked(
                t,
                warm_start,
                {"inputs": network_inputs},
                residual_tolerance=self.solve_policy.residual_tolerance,
                max_nfev=self.solve_policy.max_nfev,
            )
        return self.network_problem.solve_limited(t, warm_start, {"inputs": network_inputs}, max_nfev=self.solve_policy.max_nfev)

    def _direct_network_solution(
        self,
        t: float,
        warm_start: WarmStart | np.ndarray | None,
        network_inputs: Mapping[str, Any],
    ) -> NetworkSolution:
        if isinstance(warm_start, WarmStart):
            z = warm_start.z.copy()
        elif warm_start is None:
            z = self.Z0.copy()
        else:
            z = np.asarray(warm_start, dtype=float).copy()
        z = self._precondition_guess(t, z, network_inputs)
        z = self._sync_state_owned_algebraics(z, network_inputs)
        values = self.network_problem.values_from_z(z)
        evaluated: dict[str, Any] = {}
        for _ in range(2):
            evaluated = dict(self.network_problem._evaluator(t, values, {"inputs": network_inputs}))  # type: ignore[attr-defined]
            changed = self._promote_direct_targets(values, evaluated)
            if not changed:
                break
        evaluated = dict(self.network_problem._evaluator(t, values, {"inputs": network_inputs}))  # type: ignore[attr-defined]
        residual_vector = np.asarray(
            [float(evaluated.get(name, 0.0)) for name in self.network_problem.residual_names],
            dtype=float,
        )
        residuals = {
            name: float(residual_vector[i])
            for i, name in enumerate(self.network_problem.residual_names)
        }
        normalized = {
            name: float(residual_vector[i] / self.network_problem.residual_scales[i])
            for i, name in enumerate(self.network_problem.residual_names)
        }
        solution = NetworkSolution(
            z=z,
            values={
                **values,
                **{
                    key: float(value)
                    for key, value in evaluated.items()
                    if key not in self.network_problem.residual_names and _is_number(value)
                },
            },
            residuals=residuals,
            normalized_residuals=normalized,
            success=True,
            message="preconditioned direct algebraic evaluation",
        )
        if isinstance(warm_start, WarmStart):
            warm_start.update(solution)
        return solution

    def _promote_direct_targets(self, values: dict[str, float], evaluated: Mapping[str, Any]) -> bool:
        writable = set(self.algebraic_names)
        changed = False
        for key, value in evaluated.items():
            if not _is_number(value) or not key.endswith("_target"):
                continue
            target = key[: -len("_target")]
            if target in writable:
                new_value = float(value)
                old_value = float(values.get(target, new_value))
                if abs(new_value - old_value) > 1.0e-10 * max(abs(old_value), abs(new_value), 1.0):
                    changed = True
                values[target] = new_value
        return changed

    def _propagate_direct_hydraulics(self, values: dict[str, float]) -> bool:
        changed = False
        for _ in range(2):
            pass_changed = False
            for component in self.loaded.engine.components.values():
                pass_changed = self._sync_direct_component_flow(values, component) or pass_changed
            for connection in self.loaded.engine.connections:
                if connection.domain != "fluid":
                    continue
                source = connection.source
                target = connection.target
                source_mdot = f"{source}.mdot"
                target_mdot = f"{target}.mdot"
                if target_mdot in values and source_mdot in values:
                    pass_changed = _assign_if_changed(values, source_mdot, float(values[target_mdot])) or pass_changed
                elif source_mdot in values and target_mdot in values:
                    pass_changed = _assign_if_changed(values, target_mdot, float(values[source_mdot])) or pass_changed
            for component in self.loaded.engine.components.values():
                pass_changed = self._sync_direct_component_flow(values, component) or pass_changed
            changed = pass_changed or changed
            if not pass_changed:
                break
        return changed

    def _sync_direct_component_flow(self, values: dict[str, float], component: Any) -> bool:
        name = component.name
        component_type = component.type
        changed = False
        if component_type == "FlowSplitter":
            mode = str(component.parameters.get("split_mode", component.parameters.get("mode", "fixed_fraction"))).lower()
            split = float(component.parameters.get("split_fraction", 0.5))
            inlet = f"{name}.inlet.mdot"
            out_a = f"{name}.outlet_a.mdot"
            out_b = f"{name}.outlet_b.mdot"
            if mode in {"hydraulic", "pressure", "demand", "downstream"}:
                if inlet in values and out_b in values:
                    secondary = min(max(float(values[out_b]), 0.0), max(float(values[inlet]), 0.0))
                    changed = _assign_if_changed(values, out_b, secondary) or changed
                    changed = _assign_if_changed(values, out_a, float(values[inlet]) - secondary) or changed
                elif out_a in values and out_b in values:
                    changed = _assign_if_changed(values, inlet, float(values[out_a]) + float(values[out_b])) or changed
            elif inlet in values:
                changed = _assign_if_changed(values, out_a, split * float(values[inlet])) or changed
                changed = _assign_if_changed(values, out_b, (1.0 - split) * float(values[inlet])) or changed
        elif component_type in {"Pipe", "Valve", "MassFlowInjector", "OrificeCompressible"}:
            ports = [
                port
                for port, domain in component_spec(component_type).ports.items()
                if str(domain).startswith("fluid")
            ]
            mdot = values.get(f"{name}.mdot")
            if mdot is None:
                for port in reversed(ports):
                    candidate = values.get(f"{name}.{port}.mdot")
                    if _is_number(candidate):
                        mdot = float(candidate)
                        break
            if _is_number(mdot):
                changed = _assign_if_changed(values, f"{name}.mdot", float(mdot)) or changed
                for port in ports:
                    changed = _assign_if_changed(values, f"{name}.{port}.mdot", float(mdot)) or changed
        return changed

    def _sync_state_owned_algebraics(self, z: np.ndarray, network_inputs: Mapping[str, Any]) -> np.ndarray:
        synced = np.asarray(z, dtype=float).copy()
        for name in self.state_names:
            if name not in self.algebraic_names:
                continue
            value = network_inputs.get(name)
            if not _is_number(value):
                continue
            synced[self.algebraic_names.index(name)] = float(value)
        return synced

    def _precondition_warm_start(
        self,
        t: float,
        warm_start: WarmStart | np.ndarray | None,
        network_inputs: Mapping[str, Any],
    ) -> WarmStart | np.ndarray | None:
        if self.solve_policy.preconditioner in {"none", "off", "false"}:
            return warm_start
        if isinstance(warm_start, WarmStart):
            warm_start.z = self._precondition_guess(t, warm_start.z, network_inputs)
            return warm_start
        base = self.Z0 if warm_start is None else np.asarray(warm_start, dtype=float)
        return self._precondition_guess(t, base, network_inputs)

    def _precondition_guess(
        self,
        t: float,
        z: np.ndarray,
        network_inputs: Mapping[str, Any],
    ) -> np.ndarray:
        if self.solve_policy.preconditioner in {"none", "off", "false"}:
            return np.asarray(z, dtype=float)
        return precondition_algebraic_guess(
            self.loaded,
            self.network_problem,
            t,
            np.asarray(z, dtype=float),
            {"inputs": network_inputs},
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


def _dae_solve_policy(loaded: LoadedAnalysisConfig, execution_plan: ExecutionPlan) -> DAESolvePolicy:
    analysis = loaded.analysis_config.analysis
    raw = analysis.get("solve_policy", {})
    if not isinstance(raw, Mapping):
        raw = {}
    diagnostic_default = execution_plan.analysis_type == "port_network_diagnostics"
    allow_non_square = bool(raw.get("allow_non_square", raw.get("diagnostic", diagnostic_default)))
    checked = bool(raw.get("checked", not diagnostic_default or not allow_non_square))
    return DAESolvePolicy(
        allow_non_square=allow_non_square,
        checked=checked,
        residual_tolerance=float(raw.get("residual_tolerance", raw.get("tolerance", 1.0e-8))),
        max_nfev=int(raw["max_nfev"]) if raw.get("max_nfev") is not None else None,
        strict_sources=bool(raw.get("strict_sources", raw.get("production", False))),
        corrector=str(raw.get("corrector", raw.get("corrector_mode", "none"))),
        corrector_iterations=int(raw.get("corrector_iterations", raw.get("iterations", 1))),
        preconditioner=str(raw.get("preconditioner", "none")).lower(),
        algebraic_solver=str(raw.get("algebraic_solver", raw.get("solver", "nonlinear"))).lower(),
    )


def _unique_sorted_clipped(values: np.ndarray, start: float, end: float) -> np.ndarray:
    clipped = sorted({float(value) for value in values if start - 1.0e-12 <= float(value) <= end + 1.0e-12})
    if not clipped:
        return np.zeros(0, dtype=float)
    result: list[float] = []
    for value in clipped:
        if value < start and np.isclose(value, start, rtol=0.0, atol=1.0e-12):
            value = start
        if value > end and np.isclose(value, end, rtol=0.0, atol=1.0e-12):
            value = end
        if not result or not np.isclose(value, result[-1], rtol=0.0, atol=1.0e-12):
            result.append(float(value))
    return np.asarray(result, dtype=float)


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


def _validate_strict_full_port_sources(loaded: LoadedAnalysisConfig) -> None:
    incoming = {(connection.target, connection.domain) for connection in loaded.engine.connections}
    outgoing = {(connection.source, connection.domain) for connection in loaded.engine.connections}
    boundary_values = evaluate_boundary_conditions(loaded.boundary_conditions, 0.0) if loaded.boundary_conditions is not None else {}
    missing: list[str] = []
    for component in loaded.engine.components.values():
        spec = component_spec(component.type)
        for port_name, domain in spec.ports.items():
            if domain not in {"fluid_in", "fluid_out"}:
                continue
            endpoint = f"{component.name}.{port_name}"
            has_connection = (endpoint, "fluid") in incoming if domain == "fluid_in" else (endpoint, "fluid") in outgoing
            has_boundary = any(
                path == endpoint or path.startswith(f"{endpoint}.")
                for path in boundary_values
            )
            if has_connection or has_boundary:
                continue
            if _is_optional_alias_port(component.type, port_name, outgoing, incoming, boundary_values, component.name, domain):
                continue
            if component.type == "BoundarySource" and domain == "fluid_out":
                continue
            if component.type == "BoundarySink" and domain == "fluid_in":
                continue
            missing.append(endpoint)
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(f"strict full-port source validation failed; unconnected fluid endpoint(s): {joined}")


def _is_optional_alias_port(
    component_type: str,
    port_name: str,
    outgoing: set[tuple[str, str]],
    incoming: set[tuple[str, str]],
    boundary_values: Mapping[str, Any],
    component_name: str,
    domain: str,
) -> bool:
    if component_type not in {"CombustionChamber", "Preburner", "GasGenerator"} or domain != "fluid_in":
        return False
    aliases = {
        "lox_inlet": ("ox_inlet",),
        "ox_inlet": ("lox_inlet",),
        "inlet": ("fuel_inlet", "ox_inlet", "lox_inlet"),
    }.get(port_name)
    if aliases is None:
        return False
    for alias in aliases:
        endpoint = f"{component_name}.{alias}"
        if (endpoint, "fluid") in incoming or (endpoint, "fluid") in outgoing:
            return True
        if any(path == endpoint or path.startswith(f"{endpoint}.") for path in boundary_values):
            return True
    return False


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


def _assign_if_changed(values: dict[str, float], key: str, value: float) -> bool:
    old = values.get(key)
    values[key] = float(value)
    if not _is_number(old):
        return True
    return abs(float(value) - float(old)) > 1.0e-10 * max(abs(float(value)), abs(float(old)), 1.0)


def _largest_abs(values: Mapping[str, float]) -> tuple[str, float]:
    if not values:
        return "", 0.0
    name = max(values, key=lambda key: abs(float(values[key])))
    return name, float(values[name])
