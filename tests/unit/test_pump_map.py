from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest


def _component():
    from atha.config.schema import ComponentConfig

    return ComponentConfig(
        name="lox_pump",
        type="Pump",
        parameters={
            "diameter": 0.145,
            "pump_map": {
                "mdot_design": 30.5,
                "dP_design": 13.0e6,
                "speed_design": 32000.0,
                "efficiency_design": 0.74,
                "rho_design": 1140.0,
            },
        },
    )


def test_pump_head_contract_uses_phi_psi_map():
    from atha.components.residuals import PumpHeadContract, ResidualEvaluationContext

    diameter = 0.145
    rho = 1140.0
    omega = 32000.0 * 2.0 * math.pi / 60.0
    mdot = 30.5
    delta_p = 13.0e6
    psi = delta_p / (rho * omega**2 * diameter**2)

    head_map = MagicMock()
    head_map.evaluate.return_value = {"psi": psi}
    efficiency_map = MagicMock()
    efficiency_map.evaluate.return_value = {"eta": 0.74}

    residuals = PumpHeadContract().evaluate(
        _component(),
        ResidualEvaluationContext(
            z={"lox_pump.delta_P": delta_p},
            inputs={"lox_pump.shaft.omega": omega, "lox_pump.mdot": mdot, "lox_pump.inlet.rho": rho},
            model={
                "lox_pump.map.head_map": head_map,
                "lox_pump.map.head_map.output": "psi",
                "lox_pump.map.efficiency_map": efficiency_map,
                "lox_pump.map.efficiency_map.output": "eta",
            },
        ),
    )

    assert residuals["lox_pump.delta_P_residual"] == pytest.approx(0.0, abs=1.0)
    assert residuals["lox_pump.efficiency"] == pytest.approx(0.74)
    assert head_map.evaluate.call_args.args[0]["phi"] == pytest.approx(mdot / (rho * omega * diameter**3))


def test_pump_head_contract_falls_back_to_affinity_without_map():
    from atha.components.residuals import PumpHeadContract, ResidualEvaluationContext

    omega_design = 32000.0 * 2.0 * math.pi / 60.0
    omega = omega_design * 0.8
    delta_p = 13.0e6 * 0.8**2

    residuals = PumpHeadContract().evaluate(
        _component(),
        ResidualEvaluationContext(
            z={"lox_pump.delta_P": delta_p},
            inputs={"lox_pump.shaft.omega": omega, "lox_pump.mdot": 30.5, "lox_pump.inlet.rho": 1140.0},
            model={},
        ),
    )

    assert residuals["lox_pump.delta_P_residual"] == pytest.approx(0.0, abs=1.0)


def test_pump_compute_outputs_adds_enthalpy_rise():
    from atha.components.pump import Pump, PumpMap

    fluid = MagicMock()
    inlet_state = MagicMock()
    inlet_state.rho = 1140.0
    inlet_state.T = 90.0
    outlet_state = MagicMock()
    outlet_state.T = 93.0
    fluid.state_from_Ph.side_effect = [inlet_state, outlet_state]
    pump_map = PumpMap(mdot_design=30.5, dP_design=13.0e6, omega_design=3351.0, eta_design=0.74)
    pump = Pump("p", diameter=0.145, pump_map=pump_map, fluid=fluid)

    outputs = pump.compute_outputs(
        0.0,
        {},
        {
            "shaft.omega": 32000.0 * 2.0 * math.pi / 60.0,
            "inlet.P": 1.0e6,
            "inlet.h": 0.0,
            "inlet.mdot": 30.5,
        },
    )

    assert outputs["outlet.h"] > 0.0
    assert outputs["h"] == outputs["outlet.h"]


def test_valve_flow_contract_uses_pressure_dependent_cda_map():
    from atha.components.residuals import ResidualEvaluationContext, ValveFlowContract
    from atha.config.schema import ComponentConfig

    valve = ComponentConfig(
        name="check_valve",
        type="Valve",
        parameters={"max_area": 1.0e-4, "discharge_coeff": 0.8},
    )
    cda_map = MagicMock()
    cda_map.evaluate.return_value = {"CdA": 2.0e-4}

    residuals = ValveFlowContract().evaluate(
        valve,
        ResidualEvaluationContext(
            z={"check_valve.mdot": 2.0e-4 * math.sqrt(2.0 * 1000.0 * 1.0e5)},
            inputs={
                "check_valve.inlet.P": 2.0e5,
                "check_valve.outlet.P": 1.0e5,
                "check_valve.inlet.rho": 1000.0,
                "check_valve.position": 0.25,
            },
            model={"check_valve.map.cda_map": cda_map},
        ),
    )

    assert residuals["check_valve.mdot_residual"] == pytest.approx(0.0)
    assert residuals["check_valve.CdA"] == pytest.approx(2.0e-4)
    assert cda_map.evaluate.call_args.args[0]["inlet.P"] == pytest.approx(2.0e5)


def test_example_20_reduced_model_uses_pump_design_speed_not_initial_speed():
    from atha.analysis.gg_single_shaft import _design_model, _plant_values, _rpm_to_rad_s
    from atha.config import load_analysis_config

    loaded = load_analysis_config("examples/20_gg_single_shaft_methalox/configs/analysis.yaml")
    model = _design_model(loaded)

    assert model["shaft_omega_design"] == pytest.approx(_rpm_to_rad_s(32000.0))

    transients = {
        "main_lox_valve.position": 1.0,
        "main_methane_valve.position": 1.0,
        "lox_generator_valve.position": 0.5,
        "methane_generator_valve.position": 0.4,
    }
    low_speed = _plant_values(
        {"shaft.omega": _rpm_to_rad_s(24000.0), "chamber.P": 2.0e6, "generator.P": 4.0e6},
        transients,
        model,
    )
    design_speed = _plant_values(
        {"shaft.omega": _rpm_to_rad_s(32000.0), "chamber.P": 2.0e6, "generator.P": 4.0e6},
        transients,
        model,
    )

    assert low_speed["lox_pump.delta_P"] < design_speed["lox_pump.delta_P"]
    assert low_speed["methane_pump.delta_P"] < design_speed["methane_pump.delta_P"]
