from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from atha.assembly import EngineAssembler
from atha.config.loader import LoadedAnalysisConfig
from atha.runner.progress import SolverProgressEvent


@dataclass(frozen=True)
class AnalysisContext:
    """Shared execution context passed to analysis handlers.

    The context is the Phase-19 handoff between config loading, execution-plan
    construction, source catalogs, and analysis-specific solver implementations.
    Handlers can still accept the older ``(config_path, output_dir)`` signature,
    but new handlers should prefer this object.
    """

    loaded: LoadedAnalysisConfig
    config_path: Path
    output_dir: Path
    analysis_type: str
    mode: str
    execution_plan: Any
    registry: Any | None = None
    progress_callback: Callable[[SolverProgressEvent], None] | None = None

    @property
    def analysis(self) -> dict[str, Any]:
        return self.loaded.analysis_config.analysis

    @property
    def solver(self) -> dict[str, Any]:
        return self.loaded.analysis_config.solver

    def source_catalog(self, extra_sources=()) -> set[str]:
        return EngineAssembler(self.loaded).telemetry_sources(extra_sources)
