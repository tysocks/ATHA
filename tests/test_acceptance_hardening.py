"""Acceptance hardening for sustained thrust gates."""

from __future__ import annotations

import numpy as np

from atha.validation.acceptance import build_generic_port_acceptance_report


def test_min_powered_tail_thrust_and_final_thrust_rel() -> None:
    time = np.linspace(0.0, 10.0, 11)
    thrust = np.full(time.shape, 1.5e5)
    thrust[8:] = 2.0e3  # collapse in the powered tail
    report = build_generic_port_acceptance_report(
        case="thrust_collapse",
        time=time,
        values={
            "nozzle.thrust": thrust,
            "mdot.total": np.full(time.shape, 40.0),
            "target.mdot_total": np.full(time.shape, 40.0),
        },
        residuals={},
        tolerances={
            "min_peak_thrust": 1.0e5,
            "min_powered_tail_thrust": 1.0e5,
            "powered_tail_s": 3.0,
            "final_thrust_rel": 0.35,
            "design_thrust": 1.5e5,
            "max_normalized_residual": 1.0,
        },
        evaluation_end_s=10.0,
    )
    checks = {check.name: check for check in report.checks}
    assert checks["powered_thrust"].passed is True
    assert checks["min_powered_tail_thrust"].passed is False
    assert checks["final_thrust_tracking"].passed is False


def test_sustained_thrust_passes_when_tail_holds() -> None:
    time = np.linspace(0.0, 10.0, 11)
    thrust = np.full(time.shape, 1.5e5)
    report = build_generic_port_acceptance_report(
        case="sustained",
        time=time,
        values={"nozzle.thrust": thrust},
        residuals={},
        tolerances={
            "min_peak_thrust": 1.0e5,
            "min_powered_tail_thrust": 1.0e5,
            "final_thrust_rel": 0.35,
            "design_thrust": 1.5e5,
            "max_normalized_residual": 1.0,
        },
    )
    checks = {check.name: check for check in report.checks}
    assert checks["min_powered_tail_thrust"].passed is True
    assert checks["final_thrust_tracking"].passed is True
