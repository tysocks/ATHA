"""Verification suite registry and runner for Workstream 6.2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from atha.runner import run_config_folder
from atha.validation.acceptance import AcceptanceReport
from atha.validation.reference_checks import ReferenceCheckReport, write_reference_check_report_json


@dataclass(frozen=True)
class VerificationCaseSpec:
    """Describes one verification gate in the ATHA ladder."""

    id: str
    level: int
    config_dir: Path
    description: str
    component: str = ""
    subsystem: str = ""
    reference_notes: str = ""
    tags: tuple[str, ...] = ()
    slow: bool = False


@dataclass
class VerificationCaseResult:
    spec: VerificationCaseSpec
    acceptance_passed: bool | None = None
    acceptance_report: Path | None = None
    reference_passed: bool | None = None
    reference_report: Path | None = None
    output_dir: Path | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.errors:
            return False
        flags = [value for value in (self.acceptance_passed, self.reference_passed) if value is not None]
        return bool(flags) and all(flags)


@dataclass(frozen=True)
class VerificationSuiteReport:
    passed: bool
    results: tuple[VerificationCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "atha.verification_suite_report.v1",
            "passed": self.passed,
            "cases": [
                {
                    "id": result.spec.id,
                    "level": result.spec.level,
                    "passed": result.passed,
                    "acceptance_passed": result.acceptance_passed,
                    "reference_passed": result.reference_passed,
                    "acceptance_report": None if result.acceptance_report is None else str(result.acceptance_report),
                    "reference_report": None if result.reference_report is None else str(result.reference_report),
                    "errors": list(result.errors),
                }
                for result in self.results
            ],
        }


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSYSTEM_ROOT = REPO_ROOT / "examples" / "21_generic_port_subsystems"

VERIFICATION_CASES: tuple[VerificationCaseSpec, ...] = (
    VerificationCaseSpec(
        id="valve_pipe_volume",
        level=2,
        config_dir=SUBSYSTEM_ROOT / "valve_pipe_volume",
        description="Valve + pipe + gas-volume pressure ramp closure.",
        component="Valve",
        subsystem="valve_pipe_volume",
        reference_notes="Synthetic balance gate; future orifice oracle.",
        tags=("subsystem", "fast"),
    ),
    VerificationCaseSpec(
        id="pump_pipe_valve",
        level=2,
        config_dir=SUBSYSTEM_ROOT / "pump_pipe_valve",
        description="Pump head/power with pipe and valve branch.",
        component="Pump",
        subsystem="pump_pipe_valve",
        tags=("subsystem", "fast"),
    ),
    VerificationCaseSpec(
        id="pump_shaft_turbine",
        level=2,
        config_dir=SUBSYSTEM_ROOT / "pump_shaft_turbine",
        description="Shaft inertia response with pump load and turbine drive.",
        component="Rotor",
        subsystem="pump_shaft_turbine",
        tags=("subsystem", "fast"),
    ),
    VerificationCaseSpec(
        id="injector_chamber_nozzle",
        level=2,
        config_dir=SUBSYSTEM_ROOT / "injector_chamber_nozzle",
        description="Injector, chamber, and nozzle chain closure.",
        component="CombustionChamber",
        subsystem="injector_chamber_nozzle",
        tags=("subsystem", "fast"),
    ),
    VerificationCaseSpec(
        id="chamber_nozzle",
        level=2,
        config_dir=SUBSYSTEM_ROOT / "chamber_nozzle",
        description="Chamber pressure/temperature and nozzle thrust closure.",
        component="Nozzle",
        subsystem="chamber_nozzle",
        tags=("subsystem", "fast"),
    ),
    VerificationCaseSpec(
        id="preburner_turbine",
        level=2,
        config_dir=SUBSYSTEM_ROOT / "preburner_turbine",
        description="Preburner branch with turbine power extraction.",
        component="Preburner",
        subsystem="preburner_turbine",
        tags=("subsystem", "fast"),
    ),
    VerificationCaseSpec(
        id="regen_channel",
        level=2,
        config_dir=SUBSYSTEM_ROOT / "regen_channel",
        description="Regenerative cooling channel MVP with wall-temperature ODE.",
        component="RegenChannel",
        subsystem="regen_channel",
        tags=("subsystem", "fast", "thermal"),
    ),
    VerificationCaseSpec(
        id="lox_pump_map",
        level=1,
        config_dir=REPO_ROOT / "examples" / "23_single_lox_pump_map" / "configs",
        description="Single LOx pump map transient with regression windows.",
        component="Pump",
        reference_notes="Map table in configs/maps/lox_pump_phi_psi.csv.",
        tags=("component", "map"),
    ),
    VerificationCaseSpec(
        id="ffsc_dae_acceptance",
        level=3,
        config_dir=REPO_ROOT / "examples" / "19_ffsc_dae_acceptance" / "configs",
        description="Canonical FFSC full mission-cycle generic-port DAE profile.",
        subsystem="full_engine",
        reference_notes="docs/CANONICAL_MISSION_CASE.md",
        tags=("engine", "mission_cycle"),
        slow=True,
    ),
    VerificationCaseSpec(
        id="gg_single_shaft",
        level=3,
        config_dir=REPO_ROOT / "examples" / "20_gg_single_shaft_methalox" / "configs",
        description="Single-shaft gas-generator methalox engine profile.",
        subsystem="full_engine",
        tags=("engine",),
        slow=True,
    ),
)


def verification_cases(
    *,
    level: int | None = None,
    tag: str | None = None,
    include_slow: bool = True,
) -> list[VerificationCaseSpec]:
    cases = list(VERIFICATION_CASES)
    if level is not None:
        cases = [case for case in cases if case.level == level]
    if tag is not None:
        cases = [case for case in cases if tag in case.tags]
    if not include_slow:
        cases = [case for case in cases if not case.slow]
    return cases


def run_verification_case(
    spec: VerificationCaseSpec,
    *,
    output_dir: Path | None = None,
) -> VerificationCaseResult:
    result = VerificationCaseResult(spec=spec)
    out = output_dir or REPO_ROOT / "outputs" / "verification" / spec.id
    out.mkdir(parents=True, exist_ok=True)
    result.output_dir = out
    try:
        run_result = run_config_folder(spec.config_dir, output_dir=out)
    except Exception as exc:  # pragma: no cover - surfaced in report
        result.errors.append(str(exc))
        return result

    artifacts = run_result.artifacts
    if artifacts.acceptance_report is not None:
        result.acceptance_report = Path(artifacts.acceptance_report)
    summary = run_result.summary
    if summary is not None and hasattr(summary, "acceptance_passed"):
        result.acceptance_passed = summary.acceptance_passed
    elif result.acceptance_report is not None:
        result.acceptance_passed = _acceptance_passed_from_json(result.acceptance_report)
    reference_path = out / f"{spec.id}.reference_checks.json"
    reference_report = _reference_report_for_case(spec, run_result, out)
    if reference_report is not None:
        write_reference_check_report_json(reference_path, reference_report)
        result.reference_report = reference_path
        result.reference_passed = reference_report.passed
    return result


def run_verification_suite(
    cases: list[VerificationCaseSpec] | None = None,
    *,
    output_dir: Path | None = None,
) -> VerificationSuiteReport:
    selected = cases or verification_cases(include_slow=False)
    results = [
        run_verification_case(
            spec,
            output_dir=None if output_dir is None else output_dir / spec.id,
        )
        for spec in selected
    ]
    return VerificationSuiteReport(passed=all(result.passed for result in results), results=tuple(results))


def write_verification_suite_report(path: str | Path, report: VerificationSuiteReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _acceptance_passed_from_json(path: Path) -> bool | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "passed" in payload:
        return bool(payload["passed"])
    return None


def _reference_report_for_case(spec: VerificationCaseSpec, run_result: Any, output_dir: Path) -> ReferenceCheckReport | None:
    if spec.id == "regen_channel":
        return _regen_reference_report(output_dir)
    if spec.id == "chamber_nozzle":
        return _chamber_nozzle_reference_report(output_dir)
    return None


def _regen_reference_report(output_dir: Path) -> ReferenceCheckReport | None:
    from atha.validation.reference_checks import ReferenceCheck, ReferenceCheckReport

    csv_path = output_dir / "regen_channel.csv"
    if not csv_path.exists():
        return None
    import numpy as np

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return None
    t_wall = np.asarray(data["T_WALL"], dtype=float)
    q_hot = np.asarray(data["Q_HOT"], dtype=float)
    q_cool = np.asarray(data["Q_COOL"], dtype=float)
    checks = [
        ReferenceCheck(
            name="positive_hot_side_heat_load",
            category="regen_mvp",
            passed=float(np.nanmax(q_hot)) > 0.0,
            value=float(np.nanmax(q_hot)),
            limit=0.0,
            units="W",
            message=f"peak Q_hot={float(np.nanmax(q_hot)):.3f}",
        ),
        ReferenceCheck(
            name="positive_coolant_heat_pickup",
            category="regen_mvp",
            passed=float(np.nanmax(q_cool)) > 0.0,
            value=float(np.nanmax(q_cool)),
            limit=0.0,
            units="W",
            message=f"peak Q_cool={float(np.nanmax(q_cool)):.3f}",
        ),
        ReferenceCheck(
            name="wall_temperature_above_coolant",
            category="regen_mvp",
            passed=float(np.nanmax(t_wall)) > 150.0,
            value=float(np.nanmax(t_wall)),
            limit=150.0,
            units="K",
            message=f"peak T_wall={float(np.nanmax(t_wall)):.3f}",
        ),
    ]
    return ReferenceCheckReport(case="regen_channel", passed=all(c.passed for c in checks), checks=checks, metadata={"oracle": "thermal_sign_sanity"})


def _chamber_nozzle_reference_report(output_dir: Path) -> ReferenceCheckReport | None:
    from atha.validation.reference_checks import ReferenceCheck, ReferenceCheckReport

    csv_path = output_dir / "chamber_nozzle.csv"
    if not csv_path.exists():
        return None
    import numpy as np

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return None
    thrust = float(data["THRUST"][-1])
    mdot = float(data["MDOT_TOTAL"][-1])
    c_star = float(data["C_STAR"][-1]) if "C_STAR" in data.dtype.names else float("nan")
    thrust_scale = 3500.0
    thrust_rel = abs(thrust - thrust_scale * mdot) / max(abs(thrust_scale * mdot), 1.0e-12)
    checks = [
        ReferenceCheck(
            name="thrust_mdot_scaling",
            category="nozzle_mvp",
            passed=thrust_rel <= 0.05,
            value=thrust_rel,
            limit=0.05,
            units="rel",
            message=f"thrust={thrust:.6g}, mdot={mdot:.6g}, scale={thrust_scale}",
        )
    ]
    if np.isfinite(c_star):
        cstar_target = 1600.0
        cstar_rel = abs(c_star - cstar_target) / cstar_target
        checks.append(
            ReferenceCheck(
                name="c_star_balance",
                category="nozzle_mvp",
                passed=cstar_rel <= 0.05,
                value=cstar_rel,
                limit=0.05,
                units="rel",
                message=f"c_star={c_star:.6g}, target={cstar_target}",
            )
        )
    return ReferenceCheckReport(
        case="chamber_nozzle",
        passed=all(check.passed for check in checks),
        checks=checks,
        metadata={"oracle": "subsystem_balance_law"},
    )
