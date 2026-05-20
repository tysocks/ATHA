from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from atha.runner.context import AnalysisContext
from atha.validation.parity import (
    build_parity_report_from_files,
    parity_channel_specs_from_config,
    phase_windows_from_config,
    write_parity_delta_csv,
    write_parity_report_json,
)


@dataclass
class ParityAnalysisSummary:
    config_path: Path
    solver_status: str
    reference_config: Path
    candidate_config: Path
    reference_csv: Path
    candidate_csv: Path
    parity_report: Path
    parity_delta_csv: Path
    parity_passed: bool


def run_parity_analysis(context: AnalysisContext) -> ParityAnalysisSummary:
    """Run a reference and candidate analysis, then compare telemetry parity."""

    cfg = context.analysis.get("parity", context.analysis)
    if not isinstance(cfg, Mapping):
        raise ValueError("parity analysis requires analysis.parity mapping")
    reference_cfg = _resolve_path(context.config_path, cfg.get("reference", cfg.get("reference_config")))
    candidate_cfg = _resolve_path(context.config_path, cfg.get("candidate", cfg.get("candidate_config")))
    if reference_cfg is None or candidate_cfg is None:
        raise ValueError("parity analysis requires reference and candidate config paths")

    from atha.runner.config_runner import run_config_folder

    reference_output = context.output_dir / str(cfg.get("reference_output_dir", "reference"))
    candidate_output = context.output_dir / str(cfg.get("candidate_output_dir", "candidate"))
    reference_output.mkdir(parents=True, exist_ok=True)
    candidate_output.mkdir(parents=True, exist_ok=True)
    reference = run_config_folder(reference_cfg, output_dir=reference_output)
    candidate = run_config_folder(candidate_cfg, output_dir=candidate_output)
    if reference.csv is None:
        raise ValueError(f"reference run did not produce CSV: {reference_cfg}")
    if candidate.csv is None:
        raise ValueError(f"candidate run did not produce CSV: {candidate_cfg}")

    channels = parity_channel_specs_from_config(cfg.get("channels"))
    if not channels:
        raise ValueError("parity analysis requires at least one channel")
    windows = phase_windows_from_config(cfg.get("windows")) or _windows_from_execution_plan(context)
    case = str(cfg.get("case", context.loaded.analysis_config.name))
    report = build_parity_report_from_files(
        reference.csv,
        candidate.csv,
        case=case,
        channels=channels,
        windows=windows,
        time_column=str(cfg.get("time_column", "TIME")),
        metadata={
            "reference_config": str(reference_cfg),
            "candidate_config": str(candidate_cfg),
            "reference_analysis_type": reference.analysis_type,
            "candidate_analysis_type": candidate.analysis_type,
        },
    )
    report_path = write_parity_report_json(context.output_dir / str(cfg.get("report", "parity_report.json")), report)
    delta_path = write_parity_delta_csv(
        context.output_dir / str(cfg.get("delta_csv", "parity_delta.csv")),
        reference.csv,
        candidate.csv,
        channels=channels,
        windows=windows,
        time_column=str(cfg.get("time_column", "TIME")),
    )
    return ParityAnalysisSummary(
        config_path=context.config_path,
        solver_status="completed parity analysis",
        reference_config=reference_cfg,
        candidate_config=candidate_cfg,
        reference_csv=reference.csv,
        candidate_csv=candidate.csv,
        parity_report=report_path,
        parity_delta_csv=delta_path,
        parity_passed=report.passed,
    )


def _resolve_path(config_path: Path, raw: object) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    if path.is_dir():
        path = path / "analysis.yaml"
    return path.resolve()


def _windows_from_execution_plan(context: AnalysisContext):
    from atha.validation.parity import PhaseWindow

    return [
        PhaseWindow(name=phase.name or f"phase_{index}", start_s=phase.start_s, end_s=phase.end_s)
        for index, phase in enumerate(context.execution_plan.phases)
    ]
