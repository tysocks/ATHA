from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atha.runner.artifacts import RunArtifacts


@dataclass
class RunResult:
    """Standard result wrapper returned by config-folder runs."""

    name: str
    analysis_type: str
    config_path: Path
    csv: Path | None = None
    plot: Path | None = None
    artifacts: RunArtifacts = field(default_factory=RunArtifacts)
    summary: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_summary(self) -> Any:
        if self.summary is None:
            raise ValueError("RunResult does not contain a summary object")
        return self.summary

    def artifact_paths(self) -> dict[str, Path]:
        return self.artifacts.as_dict()
