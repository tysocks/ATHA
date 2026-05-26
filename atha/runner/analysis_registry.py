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
            type_name="port_network_diagnostics",
            handler=_run_port_network_diagnostics,
            mode="steady",
            description="Generic automatic port-variable network diagnostics and algebraic solve.",
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="steady",
            handler=_run_generic_steady,
            mode="steady",
            description="Generic steady port-network trim/diagnostics.",
            accepts_context=True,
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="profile",
            handler=_run_generic_profile,
            mode="profile",
            description="Generic DAE profile execution through the universal runner.",
            accepts_context=True,
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="linearization",
            handler=_run_generic_linearization,
            mode="linearization",
            description="Generic finite-difference DAE linearization.",
            accepts_context=True,
        )
    )
    registry.register(
        AnalysisSpec(
            type_name="parity",
            handler=_run_parity_analysis,
            mode="validation",
            description="Run reference/candidate configs and write transient parity reports.",
            accepts_context=True,
        )
    )
    return registry


def _run_port_network_diagnostics(config_path: Path, output_dir: Path) -> object:
    from atha.analysis.port_network import run_port_network_diagnostics

    return run_port_network_diagnostics(config_path, output_dir=output_dir)


def _run_generic_steady(context: AnalysisContext) -> object:
    from atha.analysis.generic_modes import run_generic_steady

    return run_generic_steady(context)


def _run_generic_profile(context: AnalysisContext) -> object:
    from atha.analysis.generic_modes import run_generic_profile

    return run_generic_profile(context)


def _run_generic_linearization(context: AnalysisContext) -> object:
    from atha.analysis.generic_modes import run_generic_linearization

    return run_generic_linearization(context)


def _run_parity_analysis(context: AnalysisContext) -> object:
    from atha.analysis.parity_mode import run_parity_analysis

    return run_parity_analysis(context)




DEFAULT_ANALYSIS_REGISTRY = default_analysis_registry()
