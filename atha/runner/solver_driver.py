from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from atha.config.loader import LoadedAnalysisConfig
from atha.config.schedules import collect_config_breakpoints
from atha.runner.analysis_registry import AnalysisRegistry
from atha.runner.context import AnalysisContext
from atha.runner.progress import SolverProgressEvent


@dataclass(frozen=True)
class SolverDriverResult:
    summary: object
    analysis_type: str
    mode: str
    execution_plan: "ExecutionPlan"


@dataclass(frozen=True)
class IntegrationOptions:
    method: str = "Radau"
    rtol: float = 1.0e-7
    atol: float = 1.0e-6
    max_step: float | None = None
    segment_at_samples: bool = False
    segment_at_controller_samples: bool = True
    per_state: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class StateMode:
    path: str
    mode: str
    value: float | None = None


@dataclass(frozen=True)
class ExecutionPhase:
    start_s: float
    end_s: float
    name: str = ""
    advance_when: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    analysis_type: str
    mode: str
    time_start_s: float
    time_end_s: float
    phases: list[ExecutionPhase]
    integration: IntegrationOptions
    state_modes: dict[str, StateMode]
    trim_enabled: bool = False
    recovery: dict[str, Any] = field(default_factory=dict)


class SolverDriver:
    """Numerical execution layer behind the config runner.

    The driver currently delegates to registered analysis handlers. It is the
    stable place for upcoming trim, phase-splitting, recovery, and linearization
    logic without expanding ConfigFolderRunner again.
    """

    def __init__(self, registry: AnalysisRegistry) -> None:
        self.registry = registry

    def run(
        self,
        loaded: LoadedAnalysisConfig,
        config_path: Path,
        output_dir: Path,
        *,
        progress_callback: Callable[[SolverProgressEvent], None] | None = None,
    ) -> SolverDriverResult:
        analysis_type = str(loaded.analysis_config.analysis.get("type", ""))
        spec = self.registry.get(analysis_type)
        execution_plan = self.build_execution_plan(loaded, analysis_type, spec.mode)
        context = AnalysisContext(
            loaded=loaded,
            config_path=config_path,
            output_dir=output_dir,
            analysis_type=analysis_type,
            mode=spec.mode,
            execution_plan=execution_plan,
            registry=self.registry,
            progress_callback=progress_callback,
        )
        summary = self.registry.run_context(context)
        if hasattr(summary, "__dict__"):
            summary.execution_plan = execution_plan
            summary.analysis_context = context
        return SolverDriverResult(summary=summary, analysis_type=analysis_type, mode=spec.mode, execution_plan=execution_plan)

    def build_execution_plan(self, loaded: LoadedAnalysisConfig, analysis_type: str, mode: str) -> ExecutionPlan:
        analysis = loaded.analysis_config.analysis
        time_cfg = analysis.get("time", {})
        t_start = float(time_cfg.get("start_s", 0.0))
        t_end = float(time_cfg.get("end_s", t_start))
        phases = _execution_phases(time_cfg, loaded, t_start, t_end)
        return ExecutionPlan(
            analysis_type=analysis_type,
            mode=mode,
            time_start_s=t_start,
            time_end_s=t_end,
            phases=phases,
            integration=_integration_options(loaded),
            state_modes=_state_modes(analysis.get("state_modes", {})),
            trim_enabled=bool(analysis.get("trim", {}).get("enabled", False)) if isinstance(analysis.get("trim", {}), dict) else False,
            recovery=dict(analysis.get("recovery", {})) if isinstance(analysis.get("recovery", {}), dict) else {},
        )


def _integration_options(loaded: LoadedAnalysisConfig) -> IntegrationOptions:
    solver_cfg = loaded.analysis_config.solver.get("transient", {})
    integration_cfg = loaded.analysis_config.analysis.get("integration", {})
    if not isinstance(integration_cfg, dict):
        integration_cfg = {}
    per_state = integration_cfg.get("per_state", {})
    return IntegrationOptions(
        method=str(integration_cfg.get("method", solver_cfg.get("method", "Radau"))),
        rtol=float(integration_cfg.get("rtol", solver_cfg.get("rtol", 1.0e-7))),
        atol=float(integration_cfg.get("atol", solver_cfg.get("atol", 1.0e-6))),
        max_step=_optional_float(integration_cfg.get("max_step", solver_cfg.get("max_step"))),
        segment_at_samples=bool(integration_cfg.get("segment_at_samples", False)),
        segment_at_controller_samples=bool(integration_cfg.get("segment_at_controller_samples", True)),
        per_state={str(key): dict(value) for key, value in per_state.items()} if isinstance(per_state, dict) else {},
    )


def _state_modes(raw: Any) -> dict[str, StateMode]:
    if not isinstance(raw, dict):
        return {}
    modes: dict[str, StateMode] = {}
    valid = {"active", "inactive", "fixed", "steady_state"}
    for path, spec in raw.items():
        if isinstance(spec, str):
            mode = spec
            value = None
        elif isinstance(spec, dict):
            mode = str(spec.get("mode", "active"))
            value = spec.get("value")
        else:
            raise ValueError(f"state mode for '{path}' must be a string or mapping")
        if mode not in valid:
            raise ValueError(f"state mode for '{path}' must be one of {sorted(valid)}, got {mode!r}")
        modes[str(path)] = StateMode(path=str(path), mode=mode, value=float(value) if value is not None else None)
    return modes


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _execution_phases(
    time_cfg: Any,
    loaded: LoadedAnalysisConfig,
    t_start: float,
    t_end: float,
) -> list[ExecutionPhase]:
    configured = time_cfg.get("phases", []) if isinstance(time_cfg, dict) else []
    if isinstance(configured, list) and configured:
        phases: list[ExecutionPhase] = []
        for index, phase in enumerate(configured):
            if not isinstance(phase, dict):
                raise ValueError("analysis.time.phases entries must be mappings")
            name = str(phase.get("name", f"phase_{index}"))
            start = float(phase.get("start_s", phase.get("start", t_start)))
            end = float(phase.get("end_s", phase.get("end", t_end)))
            if end <= start:
                raise ValueError(f"analysis.time.phases[{index}] end_s must be greater than start_s")
            advance_when = phase.get("advance_when")
            if advance_when is not None and not isinstance(advance_when, dict):
                raise ValueError(f"analysis.time.phases[{index}].advance_when must be a mapping")
            phases.append(
                ExecutionPhase(
                    start_s=start,
                    end_s=end,
                    name=name,
                    advance_when=dict(advance_when) if isinstance(advance_when, dict) else None,
                )
            )
        return phases
    breakpoints = collect_config_breakpoints(
        loaded.boundary_conditions,
        loaded.timings,
        loaded.operating_conditions,
        t_start=t_start,
        t_end=t_end,
    )
    return [
        ExecutionPhase(start_s=breakpoints[i], end_s=breakpoints[i + 1])
        for i in range(len(breakpoints) - 1)
        if breakpoints[i + 1] > breakpoints[i]
    ]
