from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunArtifacts:
    csv: Path | None = None
    plot: Path | None = None
    hdf5: Path | None = None
    manifest: Path | None = None
    residuals_csv: Path | None = None
    residuals_json: Path | None = None
    linearization: Path | None = None
    acceptance_report: Path | None = None
    monte_carlo_file: Path | None = None
    histogram: Path | None = None
    sweep_plot: Path | None = None

    @classmethod
    def from_summary(cls, summary: Any) -> "RunArtifacts":
        return cls(
            csv=getattr(summary, "csv", None),
            plot=getattr(summary, "plot", None),
            hdf5=getattr(summary, "hdf5", None),
            manifest=getattr(summary, "manifest", None),
            residuals_csv=getattr(summary, "residuals_csv", None),
            residuals_json=getattr(summary, "residuals_json", None),
            linearization=getattr(summary, "linearization", None),
            acceptance_report=getattr(summary, "acceptance_report", None),
            monte_carlo_file=getattr(summary, "monte_carlo_file", None),
            histogram=getattr(summary, "histogram", None),
            sweep_plot=getattr(summary, "sweep_plot", None),
        )

    def as_dict(self) -> dict[str, Path]:
        return {key: value for key, value in self.__dict__.items() if value is not None}
