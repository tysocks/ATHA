from __future__ import annotations

from pathlib import Path

from atha.config import load_analysis_config
from atha.runner.analysis_registry import DEFAULT_ANALYSIS_REGISTRY, AnalysisRegistry
from atha.runner.artifacts import RunArtifacts
from atha.runner.result import RunResult
from atha.runner.solver_driver import SolverDriver


def run_config_folder(path: str | Path, output_dir: str | Path = "outputs") -> RunResult:
    """Run an ATHA analysis from a config folder or Analysis YAML path."""

    return ConfigFolderRunner(path, output_dir=output_dir).run()


class ConfigFolderRunner:
    """Config-folder analysis runner backed by the analysis registry."""

    def __init__(
        self,
        path: str | Path,
        output_dir: str | Path = "outputs",
        registry: AnalysisRegistry | None = None,
    ) -> None:
        self.config_path = self._resolve_analysis_path(path)
        self.output_dir = Path(output_dir)
        self.registry = registry or DEFAULT_ANALYSIS_REGISTRY

    def run(self) -> RunResult:
        loaded = load_analysis_config(self.config_path)
        driver_result = SolverDriver(self.registry).run(loaded, self.config_path, self.output_dir)
        summary = driver_result.summary
        artifacts = RunArtifacts.from_summary(summary)
        return RunResult(
            name=loaded.analysis_config.name,
            analysis_type=driver_result.analysis_type,
            config_path=self.config_path,
            csv=artifacts.csv,
            plot=artifacts.plot,
            artifacts=artifacts,
            summary=summary,
            metadata={
                "analysis_mode": driver_result.mode,
                "phase_count": len(driver_result.execution_plan.phases),
                "artifacts": {key: str(value) for key, value in artifacts.as_dict().items()},
            },
        )

    @staticmethod
    def _resolve_analysis_path(path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            candidate = candidate / "analysis.yaml"
        return candidate.resolve()
