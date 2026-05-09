from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import difflib

from atha.runner.context import AnalysisContext

AnalysisHandler = Callable[[Path, Path], object]
ContextAnalysisHandler = Callable[[AnalysisContext], object]


@dataclass(frozen=True)
class AnalysisSpec:
    type_name: str
    handler: AnalysisHandler
    mode: str
    description: str = ""
    implemented: bool = True
    accepts_context: bool = False


class AnalysisRegistry:
    """Registry mapping YAML analysis.type strings to runner handlers."""

    def __init__(self) -> None:
        self._specs: dict[str, AnalysisSpec] = {}

    def register(self, spec: AnalysisSpec) -> None:
        if spec.type_name in self._specs:
            raise ValueError(f"analysis type '{spec.type_name}' is already registered")
        self._specs[spec.type_name] = spec

    def get(self, analysis_type: str) -> AnalysisSpec:
        try:
            return self._specs[analysis_type]
        except KeyError as exc:
            suggestions = difflib.get_close_matches(analysis_type, self.known_types(), n=3)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(
                f"Unsupported analysis type: {analysis_type!r}. "
                f"Supported: {', '.join(self.known_types())}.{suggestion_text} "
                "Add a new AnalysisSpec in atha.runner.analysis_registry.default_analysis_registry() "
                "or use a registered analysis.type in analysis.yaml."
            ) from exc

    def run(self, analysis_type: str, config_path: Path, output_dir: Path) -> object:
        spec = self.get(analysis_type)
        if not spec.implemented:
            raise NotImplementedError(f"analysis type '{analysis_type}' is registered but not implemented")
        return spec.handler(config_path, output_dir)

    def run_context(self, context: AnalysisContext) -> object:
        spec = self.get(context.analysis_type)
        if not spec.implemented:
            raise NotImplementedError(f"analysis type '{context.analysis_type}' is registered but not implemented")
        if spec.accepts_context:
            return spec.handler(context)  # type: ignore[misc]
        return spec.handler(context.config_path, context.output_dir)

    def known_types(self) -> list[str]:
        return sorted(self._specs)

    def implemented_types(self) -> list[str]:
        return sorted(type_name for type_name, spec in self._specs.items() if spec.implemented)


def default_analysis_registry() -> AnalysisRegistry:
    registry = AnalysisRegistry()
    registry.register(
        AnalysisSpec(
            type_name="valve_volume_transient",
            handler=_run_valve_volume_transient,
            mode="transient",
            description="Single valve feeding a gas volume with outlet inertia.",
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="two_valve_transient_chain",
            handler=_run_pressure_fed_tca,
            mode="transient",
            description="Pressure-fed two-leg valve/pipe/injector/chamber/nozzle chain.",
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="tca_propellant_valve_transient",
            handler=_run_pressure_fed_tca,
            mode="transient",
            description="Methalox TCA valve transient using the pressure-fed network solver.",
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="tca_mdot_controller",
            handler=_run_pressure_fed_tca,
            mode="transient",
            description="Methalox TCA with operating-condition mass-flow controller.",
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="ffsc_dae_transient",
            handler=_run_ffsc_dae_acceptance,
            mode="transient",
            description="Reduced-order FFSC DAE acceptance transient.",
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="nominal_mc_sweep",
            handler=_run_nominal_mc_sweep,
            mode="sweep",
            description="Gas-generator nominal solve with Monte Carlo and speed sweep.",
        )
    )
    return registry


def _run_valve_volume_transient(config_path: Path, output_dir: Path) -> object:
    from atha.examples.valve_volume import run_valve_volume_profile

    return run_valve_volume_profile(config_path, output_dir=output_dir)


def _run_pressure_fed_tca(config_path: Path, output_dir: Path) -> object:
    from atha.analysis.pressure_fed import run_pressure_fed_tca

    return run_pressure_fed_tca(config_path, output_dir=output_dir)


def _run_ffsc_dae_acceptance(config_path: Path, output_dir: Path) -> object:
    from atha.analysis.ffsc_acceptance import run_ffsc_dae_transient

    return run_ffsc_dae_transient(config_path, output_dir=output_dir)


def _run_nominal_mc_sweep(config_path: Path, output_dir: Path) -> object:
    from atha.analysis.gg_mc_sweep import run_nominal_mc_sweep

    return run_nominal_mc_sweep(config_path, output_dir=output_dir)


DEFAULT_ANALYSIS_REGISTRY = default_analysis_registry()
