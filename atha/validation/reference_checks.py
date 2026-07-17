"""Analytical reference checks for component MVP verification (Workstream 6.2).

These helpers provide closed-form or literature-style oracles for comparing ATHA
telemetry against expected physics without requiring an external solver.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from atha.output.comparison import compare_time_series, load_time_series_csv, write_comparison_report_json


@dataclass(frozen=True)
class ReferenceCheck:
    name: str
    category: str
    passed: bool
    value: float
    limit: float
    units: str = ""
    message: str = ""


@dataclass(frozen=True)
class ReferenceCheckReport:
    case: str
    passed: bool
    checks: list[ReferenceCheck]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "atha.reference_check_report.v1",
            "case": self.case,
            "passed": self.passed,
            "checks": [check.__dict__ for check in self.checks],
            "metadata": self.metadata,
        }


def orifice_mdot(*, cda: float, rho: float, delta_p: float) -> float:
    """Incompressible orifice mass flow [kg/s]."""

    return max(float(cda), 0.0) * math.sqrt(max(2.0 * max(float(rho), 0.0) * max(float(delta_p), 0.0), 0.0))


def nozzle_thrust(*, thrust_coefficient: float, throat_area: float, chamber_pressure: float, ambient_pressure: float) -> float:
    """Vacuum-style thrust from chamber pressure and throat area [N]."""

    return float(thrust_coefficient) * float(throat_area) * max(float(chamber_pressure) - float(ambient_pressure), 0.0)


def characteristic_velocity(*, gamma: float, gas_r: float, temperature: float) -> float:
    """Ideal-gas characteristic velocity c* [m/s]."""

    g = max(float(gamma), 1.001)
    return math.sqrt(max(float(gas_r) * float(temperature) / g, 0.0)) * (
        (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    )


def pump_head_affinity(*, head_design: float, speed_ratio: float) -> float:
    """Pump head scales with the square of shaft speed."""

    return float(head_design) * float(speed_ratio) ** 2


def pump_flow_affinity(*, mdot_design: float, speed_ratio: float) -> float:
    """Pump volumetric/mass flow scales linearly with shaft speed."""

    return float(mdot_design) * float(speed_ratio)


def regen_wall_temperature_rise(*, q_hot: float, q_cool: float, wall_mass: float, wall_cp: float, dt: float) -> float:
    """Lumped wall temperature increment from net heat load [K]."""

    denom = max(float(wall_mass) * max(float(wall_cp), 1.0e-12), 1.0e-12)
    return (float(q_hot) - float(q_cool)) * float(dt) / denom


def build_valve_orifice_reference_checks(
    *,
    case: str,
    cda: float,
    rho: float,
    delta_p: float,
    measured_mdot: float,
    rtol: float = 0.05,
) -> ReferenceCheckReport:
    expected = orifice_mdot(cda=cda, rho=rho, delta_p=delta_p)
    rel_error = abs(measured_mdot - expected) / max(abs(expected), 1.0e-12)
    check = ReferenceCheck(
        name="orifice_mdot",
        category="valve_mvp",
        passed=rel_error <= rtol,
        value=rel_error,
        limit=rtol,
        units="rel",
        message=f"measured={measured_mdot:.6g}, expected={expected:.6g}",
    )
    return ReferenceCheckReport(case=case, passed=check.passed, checks=[check], metadata={"oracle": "incompressible_orifice"})


def build_nozzle_reference_checks(
    *,
    case: str,
    thrust_coefficient: float,
    throat_area: float,
    chamber_pressure: float,
    ambient_pressure: float,
    measured_thrust: float,
    measured_mdot: float | None = None,
    c_star: float | None = None,
    measured_c_star: float | None = None,
    rtol: float = 0.10,
) -> ReferenceCheckReport:
    checks: list[ReferenceCheck] = []
    expected_thrust = nozzle_thrust(
        thrust_coefficient=thrust_coefficient,
        throat_area=throat_area,
        chamber_pressure=chamber_pressure,
        ambient_pressure=ambient_pressure,
    )
    thrust_rel = abs(measured_thrust - expected_thrust) / max(abs(expected_thrust), 1.0e-12)
    checks.append(
        ReferenceCheck(
            name="nozzle_thrust",
            category="nozzle_mvp",
            passed=thrust_rel <= rtol,
            value=thrust_rel,
            limit=rtol,
            units="rel",
            message=f"measured={measured_thrust:.6g}, expected={expected_thrust:.6g}",
        )
    )
    if measured_mdot is not None and c_star is not None and c_star > 0.0:
        expected_mdot = throat_area * chamber_pressure / c_star
        mdot_rel = abs(measured_mdot - expected_mdot) / max(abs(expected_mdot), 1.0e-12)
        checks.append(
            ReferenceCheck(
                name="nozzle_mdot_cst",
                category="nozzle_mvp",
                passed=mdot_rel <= rtol,
                value=mdot_rel,
                limit=rtol,
                units="rel",
                message=f"measured={measured_mdot:.6g}, expected={expected_mdot:.6g}",
            )
        )
    if measured_c_star is not None and c_star is not None:
        cstar_rel = abs(measured_c_star - c_star) / max(abs(c_star), 1.0e-12)
        checks.append(
            ReferenceCheck(
                name="nozzle_c_star",
                category="nozzle_mvp",
                passed=cstar_rel <= rtol,
                value=cstar_rel,
                limit=rtol,
                units="rel",
                message=f"measured={measured_c_star:.6g}, expected={c_star:.6g}",
            )
        )
    return ReferenceCheckReport(
        case=case,
        passed=all(check.passed for check in checks),
        checks=checks,
        metadata={"oracle": "thrust_coefficient_and_cst"},
    )


def build_pump_map_reference_checks(
    *,
    case: str,
    phi: float,
    psi: float,
    eta_expected: float,
    eta_measured: float,
    rtol: float = 0.05,
) -> ReferenceCheckReport:
    rel_error = abs(eta_measured - eta_expected) / max(abs(eta_expected), 1.0e-12)
    check = ReferenceCheck(
        name="pump_map_efficiency",
        category="pump_mvp",
        passed=rel_error <= rtol,
        value=rel_error,
        limit=rtol,
        units="rel",
        message=f"phi={phi:.6g}, psi={psi:.6g}, measured={eta_measured:.6g}, expected={eta_expected:.6g}",
    )
    return ReferenceCheckReport(case=case, passed=check.passed, checks=[check], metadata={"oracle": "pump_map_table"})


def compare_csv_against_reference(
    *,
    case: str,
    actual_csv: str | Path,
    reference_csv: str | Path,
    channels: Mapping[str, float],
    time_column: str = "TIME",
) -> ReferenceCheckReport:
    """Compare selected telemetry channels against a reference CSV."""

    ref_time, ref_channels = load_time_series_csv(reference_csv, time_column=time_column)
    act_time, act_channels = load_time_series_csv(actual_csv, time_column=time_column)
    comparisons = compare_time_series(ref_time, ref_channels, act_time, act_channels)
    checks: list[ReferenceCheck] = []
    by_channel = {item.channel: item for item in comparisons}
    for channel, limit in channels.items():
        item = by_channel.get(channel)
        if item is None:
            checks.append(
                ReferenceCheck(
                    name=f"{channel}_present",
                    category="reference_trace",
                    passed=False,
                    value=1.0,
                    limit=0.0,
                    message=f"missing channel {channel}",
                )
            )
            continue
        checks.append(
            ReferenceCheck(
                name=f"{channel}_rmse",
                category="reference_trace",
                passed=item.rmse <= limit,
                value=item.rmse,
                limit=limit,
                units="same_as_channel",
                message=f"rmse={item.rmse:.6g}, max_error={item.max_error:.6g}",
            )
        )
    return ReferenceCheckReport(
        case=case,
        passed=all(check.passed for check in checks),
        checks=checks,
        metadata={"reference_csv": str(reference_csv), "actual_csv": str(actual_csv)},
    )


def write_reference_check_report_json(path: str | Path, report: ReferenceCheckReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_comparison_artifact(path: str | Path, comparisons) -> Path:
    return write_comparison_report_json(path, comparisons)
