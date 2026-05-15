from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_controller(kp: float = 1.0, kd: float = 1.0, ki: float = 0.0):
    return {
        "type": "pid",
        "inputs": {"target": "targets.setpoint", "measurement": "measurements.value"},
        "output": "valve.command",
        "parameters": {
            "proportional_gain": kp,
            "derivative_gain": kd,
            "integral_gain": ki,
            "lower_limit": -10.0,
            "upper_limit": 10.0,
        },
    }


def test_pid_derivative_uses_previous_sample_and_dt():
    from atha.runner.dae_execution import DAEExecutionProblem

    problem = DAEExecutionProblem.__new__(DAEExecutionProblem)
    problem._controller_period_s = 0.1
    problem._controller_hold_cache = {
        0: {"controller.valve_ctrl.error": 3.0},
    }

    result = problem._feedback_controller(
        "valve_ctrl",
        _make_controller(kp=0.0, kd=1.0),
        {"setpoint": 5.0},
        {},
        {"value": 4.0},
        {},
        {},
        sample_index=1,
    )

    assert result["controller.valve_ctrl.derivative"] == pytest.approx(-20.0)
    assert result["valve.command"] == pytest.approx(-10.0)


def test_pid_derivative_zero_at_first_sample():
    from atha.runner.dae_execution import DAEExecutionProblem

    problem = DAEExecutionProblem.__new__(DAEExecutionProblem)
    problem._controller_period_s = 0.1
    problem._controller_hold_cache = {}

    result = problem._feedback_controller(
        "valve_ctrl",
        _make_controller(kp=0.0, kd=1.0),
        {"setpoint": 5.0},
        {},
        {"value": 2.0},
        {},
        {},
        sample_index=0,
    )

    assert result["controller.valve_ctrl.derivative"] == pytest.approx(0.0)


def test_previous_error_is_not_an_ode_state():
    from atha.config.controllers import controller_state_infos

    config = MagicMock()
    config.controllers = {"ctrl": _make_controller()}
    names = [state.name for state in controller_state_infos(config)]

    assert "controller.ctrl.integral" in names
    assert not any("previous_error" in name for name in names)


def test_pid_integral_accumulates_from_previous_sample():
    from atha.config.controllers import evaluate_dynamic_controllers
    from atha.config.schema import ControllerConfig

    config = ControllerConfig(name="controllers", controllers={"ctrl": _make_controller(kp=0.0, kd=0.0, ki=1.0)})
    first = evaluate_dynamic_controllers(
        config,
        {"setpoint": 3.0},
        {},
        {"value": 1.0},
        dt=0.5,
    )
    second = evaluate_dynamic_controllers(
        config,
        {"setpoint": 3.0},
        {},
        {"value": 1.0},
        dt=0.5,
        previous_outputs=first,
    )

    assert first["controller.ctrl.integral"] == pytest.approx(1.0)
    assert second["controller.ctrl.integral"] == pytest.approx(2.0)
    assert second["valve.command"] == pytest.approx(2.0)
    assert second["controller.ctrl.integral_term"] == pytest.approx(2.0)


def test_pid_reports_p_i_d_terms():
    from atha.config.controllers import evaluate_dynamic_controllers
    from atha.config.schema import ControllerConfig

    config = ControllerConfig(name="controllers", controllers={"ctrl": _make_controller(kp=2.0, kd=0.5, ki=3.0)})
    result = evaluate_dynamic_controllers(
        config,
        {"setpoint": 3.0},
        {},
        {"value": 1.0},
        dt=0.5,
        previous_outputs={"controller.ctrl.error": 1.0, "controller.ctrl.integral": 4.0},
    )

    assert result["controller.ctrl.proportional_term"] == pytest.approx(4.0)
    assert result["controller.ctrl.integral_term"] == pytest.approx(15.0)
    assert result["controller.ctrl.derivative_term"] == pytest.approx(1.0)
    assert result["controller.ctrl.raw_command"] == pytest.approx(20.0)
