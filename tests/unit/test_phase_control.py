from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock


@dataclass(frozen=True)
class _Phase:
    name: str
    start_s: float
    end_s: float


def _make_problem(phases, controllers):
    from atha.runner.dae_execution import DAEExecutionProblem

    loaded = MagicMock()
    loaded.controllers = MagicMock()
    loaded.controllers.controllers = controllers
    loaded.controllers.path = None

    execution_plan = MagicMock()
    execution_plan.phases = phases
    execution_plan.time_end_s = phases[-1].end_s

    problem = DAEExecutionProblem.__new__(DAEExecutionProblem)
    problem.loaded = loaded
    problem.execution_plan = execution_plan
    problem._controller_period_s = 0.1
    problem._controller_hold_cache = {}
    return problem


def test_controller_inactive_outside_active_phase():
    ctrl = {
        "type": "proportional",
        "inputs": {"target": "targets.sp", "measurement": "measurements.value"},
        "output": "valve.command",
        "active_phases": ["CLC"],
        "parameters": {"gain": 1.0},
    }
    problem = _make_problem([_Phase("startup", 0.0, 3.0), _Phase("CLC", 3.0, 10.0)], {"ctrl": ctrl})

    outputs = problem._evaluate_controllers(1.0, {"sp": 2.0}, {}, {"value": 1.0}, {})

    assert outputs == {}


def test_controller_active_inside_active_phase():
    ctrl = {
        "type": "proportional",
        "inputs": {"target": "targets.sp", "measurement": "measurements.value"},
        "output": "valve.command",
        "active_phases": ["CLC"],
        "parameters": {"gain": 1.0},
    }
    problem = _make_problem([_Phase("startup", 0.0, 3.0), _Phase("CLC", 3.0, 10.0)], {"ctrl": ctrl})

    outputs = problem._evaluate_controllers(4.0, {"sp": 2.0}, {}, {"value": 1.0}, {})

    assert outputs["valve.command"] == 1.0
    assert outputs["controller.ctrl.error"] == 1.0
