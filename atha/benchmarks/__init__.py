"""Simplify ATHA benchmark helper by instrumenting through profile summary counters."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from atha.config import load_analysis_config
from atha.runner import run_config_folder

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    config_dir: Path
    description: str
    size: str  # fast | medium | slow


@dataclass
class BenchmarkResult:
    id: str
    size: str
    description: str
    wall_time_s: float
    algebraic_solve_count: int | None = None
    algebraic_solve_skip_count: int | None = None
    max_abs_normalized_residual: float | None = None
    output_bytes: int = 0
    output_dir: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id="chamber_nozzle",
        config_dir=REPO_ROOT / "examples" / "21_generic_port_subsystems" / "chamber_nozzle",
        description="Fast subsystem balance profile",
        size="fast",
    ),
    BenchmarkCase(
        id="lox_pump_map",
        config_dir=REPO_ROOT / "examples" / "23_single_lox_pump_map" / "configs",
        description="Medium pump-map transient",
        size="medium",
    ),
    BenchmarkCase(
        id="ffsc_dae_acceptance",
        config_dir=REPO_ROOT / "examples" / "19_ffsc_dae_acceptance" / "configs",
        description="Full-engine FFSC mission cycle",
        size="slow",
    ),
)


def _analysis_path(config_dir: Path) -> Path:
    if config_dir.is_file():
        return config_dir
    candidate = config_dir / "analysis.yaml"
    return candidate if candidate.exists() else config_dir


def run_benchmark_case(case: BenchmarkCase, *, output_dir: Path) -> BenchmarkResult:
    """Run one benchmark case and collect wall-time / solve metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # Validate configs load before timing the full runner path.
    load_analysis_config(_analysis_path(case.config_dir))

    started = time.perf_counter()
    result = run_config_folder(case.config_dir, output_dir=output_dir, progress=False)
    wall = time.perf_counter() - started

    summary = result.summary
    max_residual = None
    if result.artifacts.acceptance_report and Path(result.artifacts.acceptance_report).exists():
        payload = json.loads(Path(result.artifacts.acceptance_report).read_text(encoding="utf-8"))
        for check in payload.get("checks", []):
            if check.get("name") == "max_normalized_residual":
                max_residual = float(check.get("value", 0.0))
                break

    output_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    return BenchmarkResult(
        id=case.id,
        size=case.size,
        description=case.description,
        wall_time_s=wall,
        algebraic_solve_count=getattr(summary, "algebraic_solve_count", None),
        algebraic_solve_skip_count=getattr(summary, "algebraic_solve_skip_count", None),
        max_abs_normalized_residual=max_residual,
        output_bytes=output_bytes,
        output_dir=str(output_dir),
        metadata={
            "solver_source": getattr(summary, "solver_source", None),
            "acceptance_passed": getattr(summary, "acceptance_passed", None),
        },
    )


def run_benchmark_suite(
    *,
    output_dir: Path | None = None,
    include_slow: bool = False,
) -> list[BenchmarkResult]:
    root = output_dir or REPO_ROOT / "outputs" / "benchmarks"
    results: list[BenchmarkResult] = []
    for case in BENCHMARK_CASES:
        if case.size == "slow" and not include_slow:
            continue
        results.append(run_benchmark_case(case, output_dir=root / case.id))
    return results


def write_benchmark_report(path: Path, results: list[BenchmarkResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "atha.benchmark_report.v1",
        "cases": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
