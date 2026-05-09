from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from atha.output.diagnostics import (
    residual_diagnostics_from_mapping,
    write_residual_diagnostics_csv,
    write_residual_diagnostics_json,
)
from atha.output.plotting import plot_telemetry
from atha.output.telemetry import (
    build_telemetry_rows,
    write_output_manifest,
    write_telemetry_csv,
    write_telemetry_hdf5,
)


@dataclass(frozen=True)
class OutputArtifacts:
    csv: Path | None = None
    plot: Path | None = None
    hdf5: Path | None = None
    manifest: Path | None = None
    residuals_csv: Path | None = None
    residuals_json: Path | None = None


@dataclass(frozen=True)
class OutputProcessor:
    output_dir: Path
    telemetry_config: Any
    run_output: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def write(
        self,
        samples: list[Mapping[str, Any]],
        *,
        residuals: Mapping[str, float] | None = None,
        state_history: Mapping[str, np.ndarray] | None = None,
        algebraic_history: Mapping[str, np.ndarray] | None = None,
        residual_history: Mapping[str, np.ndarray] | None = None,
        boundary_history: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[OutputArtifacts, list[str], dict[str, np.ndarray]]:
        self.output_dir.mkdir(exist_ok=True)
        headers, columns = build_telemetry_rows(self.telemetry_config, samples)
        csv_path = write_telemetry_csv(self.output_dir / str(self.run_output["csv"]), headers, columns)
        plot_path = plot_telemetry(self.output_dir / str(self.run_output["plot"]), self.telemetry_config, columns)
        hdf5_path = write_telemetry_hdf5(
            self.output_dir / str(self.run_output.get("hdf5", Path(str(self.run_output["csv"])).with_suffix(".h5").name)),
            headers,
            columns,
            telemetry_config=self.telemetry_config,
            metadata=self.metadata,
            state_history=state_history,
            algebraic_history=algebraic_history,
            residual_history=residual_history,
            boundary_history=boundary_history,
        )
        residuals_csv = None
        residuals_json = None
        if residuals is not None:
            records = residual_diagnostics_from_mapping(residuals)
            stem = Path(str(self.run_output["csv"])).with_suffix("")
            residuals_csv = write_residual_diagnostics_csv(self.output_dir / f"{stem.name}_residuals.csv", records)
            residuals_json = write_residual_diagnostics_json(self.output_dir / f"{stem.name}_residuals.json", records)
        artifacts = OutputArtifacts(
            csv=csv_path,
            plot=plot_path,
            hdf5=hdf5_path,
            residuals_csv=residuals_csv,
            residuals_json=residuals_json,
        )
        manifest_path = write_output_manifest(
            self.output_dir / str(self.run_output.get("manifest", Path(str(self.run_output["csv"])).with_suffix(".manifest.json").name)),
            {
                "csv": artifacts.csv,
                "plot": artifacts.plot,
                "hdf5": artifacts.hdf5,
                "residuals_csv": artifacts.residuals_csv,
                "residuals_json": artifacts.residuals_json,
            },
            self.metadata,
        )
        artifacts = OutputArtifacts(
            csv=artifacts.csv,
            plot=artifacts.plot,
            hdf5=artifacts.hdf5,
            manifest=manifest_path,
            residuals_csv=artifacts.residuals_csv,
            residuals_json=artifacts.residuals_json,
        )
        return artifacts, headers, columns
