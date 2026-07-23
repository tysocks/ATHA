"""Level 0 analytical reference checks."""

from __future__ import annotations

import math

import pytest

from atha.validation.reference_checks import (
    build_nozzle_reference_checks,
    build_valve_orifice_reference_checks,
    characteristic_velocity,
    nozzle_thrust,
    orifice_mdot,
    pump_flow_affinity,
    pump_head_affinity,
    regen_wall_temperature_rise,
)


def test_orifice_mdot_sqrt_law() -> None:
    mdot = orifice_mdot(cda=1.0e-4, rho=1000.0, delta_p=1.0e5)
    assert mdot == pytest.approx(1.414213562, rel=1.0e-6)


def test_nozzle_thrust_pressure_difference() -> None:
    thrust = nozzle_thrust(thrust_coefficient=1.5, throat_area=0.01, chamber_pressure=5.0e6, ambient_pressure=101325.0)
    assert thrust == pytest.approx(73480.125, rel=1.0e-6)


def test_characteristic_velocity_positive() -> None:
    cstar = characteristic_velocity(gamma=1.2, gas_r=300.0, temperature=3500.0)
    assert cstar > 0.0


def test_pump_affinity_laws() -> None:
    assert pump_head_affinity(head_design=1.0e6, speed_ratio=0.8) == pytest.approx(6.4e5)
    assert pump_flow_affinity(mdot_design=10.0, speed_ratio=0.8) == pytest.approx(8.0)


def test_regen_wall_temperature_rise() -> None:
    delta = regen_wall_temperature_rise(q_hot=1.0e5, q_cool=2.0e4, wall_mass=2.0, wall_cp=500.0, dt=1.0)
    assert delta == pytest.approx(80.0)


def test_valve_orifice_reference_report_passes_on_exact_match() -> None:
    expected = orifice_mdot(cda=2.0e-4, rho=1140.0, delta_p=2.0e5)
    report = build_valve_orifice_reference_checks(
        case="unit_valve",
        cda=2.0e-4,
        rho=1140.0,
        delta_p=2.0e5,
        measured_mdot=expected,
    )
    assert report.passed


def test_nozzle_reference_report_passes_on_exact_match() -> None:
    thrust = nozzle_thrust(thrust_coefficient=1.5, throat_area=0.01, chamber_pressure=5.0e6, ambient_pressure=101325.0)
    report = build_nozzle_reference_checks(
        case="unit_nozzle",
        thrust_coefficient=1.5,
        throat_area=0.01,
        chamber_pressure=5.0e6,
        ambient_pressure=101325.0,
        measured_thrust=thrust,
    )
    assert report.passed
