# Pump Map φ/ψ, PID Derivative, and Phase-Based Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three interrelated gaps in ATHA: (1) pump maps must use dimensionless flow/head coefficients (φ/ψ), (2) the PID derivative term must be divided by dt, and (3) controllers must only activate during specified simulation phases (startup, CLC, shutdown).

**Architecture:** The three features are independent enough to implement and test sequentially. Pump map changes touch `residuals.py` + `pump.py` + example map files. PID derivative fix is a two-line bug fix in `controllers.py` and `dae_execution.py` with a state registration change. Phase control adds a `phases` field to the analysis schema and a phase-lookup in the controller evaluation loop.

**Tech Stack:** Python 3.11+, scipy, numpy, pytest, YAML, CSV

---

## Context

Examples 19 and 20 are the acceptance tests for this tool. Both must run to completion with physically correct behavior:

- **Example 19** (`examples/19_ffsc_dae_acceptance/`) — Full-flow staged combustion (FFSC) DAE transient with two shafts, two turbopumps, two preburners, and proportional controllers. Uses CSV pump maps with `speed_ratio/flow_ratio` axes today; must be converted to `phi/psi`.
- **Example 20** (`examples/20_gg_single_shaft_methalox/`) — Gas-generator single-shaft transient with PID controllers. Uses analytic constant maps today; must convert to constant `psi` (head coefficient).

**Three bugs to fix:**

1. **Pump map axes**: `PumpHeadContract.evaluate()` (`atha/components/residuals.py:252-280`) uses `speed_ratio` and `flow_ratio` to look up `pressure_rise`. ROCETs spec requires φ = ṁ/(ρ·N·D³) as the single map axis and ψ = ΔP/(ρ·N²·D²) as the head coefficient output. Efficiency map is loaded but never consulted.

2. **Outlet enthalpy**: `pump.py:208-209` hardcodes `h_out = h_in` and `T_out = T_in`. Correct formula: `h_out = h_in + ΔP/(ρ·η)`.

3. **PID derivative = 0**: `dae_execution.py:248` computes `derivative = error - previous_error` without dividing by `controller_period_s`. Additionally `previous_error` is registered as an ODE state (`controller_state_infos` in `controllers.py:186`), and its state equation `dx = error - previous_error` drives it to converge to the current error between controller sample instants—so the "previous" error is always nearly equal to the current error, giving derivative ≈ 0.

4. **No phase-based controller activation**: Controllers run unconditionally from t=0. Startup and shutdown phases require open-loop operation; only the CLC phase should run feedback controllers.

---

## File Map

| File | Change |
|---|---|
| `atha/components/residuals.py` | `PumpHeadContract.evaluate()`: use φ/ψ axes + efficiency |
| `atha/components/pump.py` | `compute_outputs()`: fix h_out and T_out using efficiency |
| `atha/config/schema.py` | Add `PhaseConfig` and `phases` list to analysis time config |
| `atha/config/controllers.py` | Remove `previous_error` ODE state; accept `active_phases`; accept `dt` in stateless path |
| `atha/runner/dae_execution.py` | Use cache for derivative; add `_current_phase()`; skip inactive controllers |
| `examples/19_ffsc_dae_acceptance/configs/maps/lox_pump_affinity.yaml` | Switch to `phi` axis, `psi`+`eta` outputs |
| `examples/19_ffsc_dae_acceptance/configs/maps/lox_pump_affinity.csv` | Regenerate with φ/ψ values |
| `examples/19_ffsc_dae_acceptance/configs/maps/methane_pump_affinity.yaml` | Same |
| `examples/19_ffsc_dae_acceptance/configs/maps/methane_pump_affinity.csv` | Same |
| `examples/19_ffsc_dae_acceptance/configs/engine.yaml` | Add `rho_design` and `diameter` to pump params |
| `examples/19_ffsc_dae_acceptance/configs/analysis.yaml` | Add `phases` section |
| `examples/19_ffsc_dae_acceptance/configs/controller.yaml` | Add `active_phases: [CLC]` to each controller |
| `examples/20_gg_single_shaft_methalox/configs/maps/lox_pump_affinity.yaml` | Switch to constant `psi` |
| `examples/20_gg_single_shaft_methalox/configs/maps/methane_pump_affinity.yaml` | Same |
| `examples/20_gg_single_shaft_methalox/configs/engine.yaml` | Add `rho_design`, `diameter` to pump params |
| `examples/20_gg_single_shaft_methalox/configs/analysis.yaml` | Add `phases` section |
| `examples/20_gg_single_shaft_methalox/configs/controller.yaml` | Add `active_phases: [CLC]` |
| `tests/unit/test_pump_map.py` | New unit tests for φ/ψ contract |
| `tests/unit/test_pid_derivative.py` | New unit tests for PID derivative |
| `tests/unit/test_phase_control.py` | New unit tests for phase activation |

---

## Design-Point φ/ψ Values (Pre-Computed)

**Example 19 — LOX pump** (D=0.145 m, ρ_design=1140 kg/m³, N_design=32000 rpm=3351 rad/s):
- φ_design = 30.5 / (1140 × 3351 × 0.145³) = **0.002619**
- ψ_design = 13.0e6 / (1140 × 3351² × 0.145²) = **0.04831**

**Example 19 — Methane pump** (D=0.105 m, ρ_design=422 kg/m³, N_design=27000 rpm=2827 rad/s):
- φ_design = 9.5 / (422 × 2827 × 0.105³) = **0.006879**
- ψ_design = 13.5e6 / (422 × 2827² × 0.105²) = **0.36292**

**Example 20 — LOX pump** (D=0.145 m, ρ_design=1140 kg/m³, N_design=32000 rpm):
- φ_design = 30.48 / 11645 = **0.002617**, ψ_design = 12.0e6 / 269137000 = **0.04460**

**Example 20 — Methane pump** (D=0.090 m, ρ_design=422 kg/m³, N_design=32000 rpm=3351 rad/s):
- D³ = 0.000729 m³
- φ_design = 9.52 / (422 × 3351 × 0.000729) = **0.009224**, ψ_design = 12.5e6 / (422 × 3351² × 0.081) = **0.32593**

---

## Task 1: PID Derivative Fix — Remove `previous_error` ODE State

**Files:**
- Modify: `atha/config/controllers.py:174-189` (`controller_state_infos`)
- Modify: `atha/runner/dae_execution.py:230-271` (`_feedback_controller`)
- Modify: `atha/runner/dae_execution.py:292-319` (`_controller_derivatives`)
- Create: `tests/unit/test_pid_derivative.py`

**Root cause:** `previous_error` is registered as an ODE state with equation  
`dx[prev_err] = error - prev_err`. This drives `prev_err → error` continuously, so by each sample instant `prev_err ≈ error` → derivative ≈ 0. Fix: track previous error through the sample cache, not the ODE.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pid_derivative.py
import pytest
from unittest.mock import MagicMock
import numpy as np


def _make_controller(kp=1.0, kd=1.0, ki=0.0):
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


class TestPIDDerivative:
    def test_derivative_nonzero_after_error_change(self):
        """Derivative term must be nonzero when error changes between samples."""
        from atha.runner.dae_execution import DAEExecutionProblem

        # Build minimal loaded config mock
        loaded = MagicMock()
        loaded.controllers = MagicMock()
        loaded.controllers.controllers = {"valve_ctrl": _make_controller(kp=0.0, kd=1.0)}
        loaded.controllers.evaluation = {"frequency_hz": 10.0}
        loaded.transients = None
        loaded.boundaries = None
        loaded.timings = None
        loaded.operating_conditions = None
        loaded.engine = MagicMock()
        loaded.engine.components = {}
        loaded.engine.connections = []

        execution_plan = MagicMock()
        execution_plan.state_modes = {}
        execution_plan.time_start_s = 0.0
        execution_plan.time_end_s = 1.0
        execution_plan.phases = []

        problem = DAEExecutionProblem.__new__(DAEExecutionProblem)
        problem.loaded = loaded
        problem.execution_plan = execution_plan
        problem._controller_period_s = 0.1  # 10 Hz
        problem._controller_hold_cache = {}
        problem.state_names = []
        problem._controller_state_indexes = {}

        targets = {"setpoint": 5.0}
        timings = {}
        states = {}

        # First sample at t=0.05 (sample index 0): error = 5 - 2 = 3
        measurements_t0 = {"value": 2.0}
        result_t0 = problem._feedback_controller(
            "valve_ctrl", _make_controller(kp=0.0, kd=1.0),
            targets, timings, measurements_t0, {}, states
        )
        # Cache sample 0 so sample 1 can find it
        problem._controller_hold_cache[0] = result_t0

        # Second sample at t=0.15 (sample index 1): error = 5 - 4 = 1
        measurements_t1 = {"value": 4.0}
        result_t1 = problem._feedback_controller(
            "valve_ctrl", _make_controller(kp=0.0, kd=1.0),
            targets, timings, measurements_t1, {}, states,
            sample_index=1,
        )

        # derivative = (1 - 3) / 0.1 = -20
        # command = kd * derivative = 1.0 * (-20) = -20, clamped to -10
        assert result_t1["controller.valve_ctrl.derivative"] == pytest.approx(-20.0, abs=1e-9)
        assert result_t1["valve_ctrl.command"] == pytest.approx(-10.0)

    def test_derivative_zero_at_first_sample(self):
        """At the very first sample there is no previous sample, derivative must be 0."""
        from atha.runner.dae_execution import DAEExecutionProblem

        problem = DAEExecutionProblem.__new__(DAEExecutionProblem)
        problem._controller_period_s = 0.1
        problem._controller_hold_cache = {}

        targets = {"setpoint": 5.0}
        timings = {}
        states = {}
        measurements = {"value": 2.0}

        result = problem._feedback_controller(
            "valve_ctrl", _make_controller(kp=0.0, kd=1.0),
            targets, timings, measurements, {}, states,
            sample_index=0,
        )
        assert result["controller.valve_ctrl.derivative"] == pytest.approx(0.0, abs=1e-9)

    def test_previous_error_not_in_state_infos(self):
        """previous_error must NOT be registered as an ODE state."""
        from atha.config.controllers import controller_state_infos

        config = MagicMock()
        config.controllers = {"ctrl": _make_controller()}
        infos = controller_state_infos(config)
        names = [i.name for i in infos]
        assert not any("previous_error" in n for n in names), (
            f"previous_error should not be an ODE state; found: {names}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_pid_derivative.py -v
```
Expected: FAIL — `_feedback_controller` missing `sample_index` parameter, `derivative` still computed as raw difference.

- [ ] **Step 3: Remove `previous_error` from ODE states in `controllers.py`**

In `atha/config/controllers.py`, remove lines 185-186:
```python
# DELETE these two lines:
if controller_type == "pid":
    states.append(ControllerStateInfo(f"controller.{name}.previous_error", float(params.get("previous_error_initial", 0.0))))
```

- [ ] **Step 4: Fix `_feedback_controller` signature and derivative in `dae_execution.py`**

Replace the `_feedback_controller` method (lines 230-271) with:

```python
def _feedback_controller(
    self,
    name: str,
    controller: Mapping[str, Any],
    targets: Mapping[str, Any],
    timings: Mapping[str, Any],
    measurements: Mapping[str, Any],
    outputs: Mapping[str, Any],
    states: Mapping[str, float],
    sample_index: int | None = None,
) -> dict[str, Any]:
    ctype = str(controller.get("type", "proportional"))
    inputs = controller["inputs"]
    params = controller.get("parameters", {})
    target = float(_lookup_signal(str(inputs["target"]), targets, timings, measurements, outputs))
    measurement = float(_lookup_signal(str(inputs["measurement"]), targets, timings, measurements, outputs))
    error = target - measurement
    integral = states.get(f"controller.{name}.integral", float(params.get("integral_initial", 0.0)))

    # Derivative: use previous sample's cached error, NOT an ODE state.
    if ctype == "pid" and self._controller_period_s:
        prev_idx = (sample_index or 0) - 1
        prev_cache = self._controller_hold_cache.get(prev_idx, {})
        previous_error = float(prev_cache.get(f"controller.{name}.error", error))
        derivative = (error - previous_error) / max(self._controller_period_s, 1e-12)
    else:
        derivative = 0.0

    raw = (
        float(params.get("bias", 0.0))
        + float(params.get("feed_forward_gain", 0.0)) * target
        + float(params.get("gain", params.get("proportional_gain", params.get("kp", 0.0)))) * error
    )
    if ctype in {"pi", "pid"}:
        raw += float(params.get("ki", params.get("integral_gain", 0.0))) * integral
    if ctype == "pid":
        raw += float(params.get("kd", params.get("derivative_gain", 0.0))) * derivative
    lower = float(params.get("lower_limit", params.get("min", -float("inf"))))
    upper = float(params.get("upper_limit", params.get("max", float("inf"))))
    command = min(max(raw, lower), upper)
    return {
        str(controller["output"]): command,
        f"controller.{name}.target": target,
        f"controller.{name}.measurement": measurement,
        f"controller.{name}.error": error,
        f"controller.{name}.command": command,
        f"controller.{name}.raw_command": raw,
        f"controller.{name}.saturated": float(command != raw),
        f"controller.{name}.integral": integral,
        f"controller.{name}.derivative": derivative,
    }
```

- [ ] **Step 5: Pass `sample_index` from `_evaluate_controllers` to `_feedback_controller`**

In `_evaluate_controllers`, find where `_feedback_controller` is called and pass `sample_index`:
```python
outputs.update(self._feedback_controller(name, single, targets, timings, measurements, outputs, states, sample_index=sample_index))
```

- [ ] **Step 6: Remove `previous_error` state derivative in `_controller_derivatives`**

In `_controller_derivatives` (lines 304-308), remove the `pid` block that updates `previous_error`:
```python
# DELETE these lines:
if ctype == "pid":
    state_name = f"controller.{name}.previous_error"
    index = self._controller_state_indexes.get(state_name)
    if index is not None:
        dx[index] = point.commands.get(f"controller.{name}.error", 0.0) - point.states.get(state_name, 0.0)
```

- [ ] **Step 7: Fix stateless `_evaluate_feedback_controller` in `controllers.py`**

The stateless path (used outside DAE loop) also has the bug. Add a `dt` parameter:

```python
def _evaluate_feedback_controller(
    name: str,
    controller: Mapping[str, Any],
    targets: Mapping[str, Any],
    timings: Mapping[str, Any],
    measurements: Mapping[str, Any],
    outputs: Mapping[str, Any],
    dt: float = 1.0,
) -> dict[str, Any]:
    controller_type = str(controller.get("type", "proportional"))
    inputs = controller["inputs"]
    params = controller.get("parameters", {})
    target = float(_lookup_signal(str(inputs["target"]), targets, timings, measurements, outputs))
    measurement = float(_lookup_signal(str(inputs["measurement"]), targets, timings, measurements, outputs))
    error = target - measurement
    bias = float(params.get("bias", 0.0))
    feed_forward = float(params.get("feed_forward_gain", 0.0)) * target
    kp = float(params.get("gain", params.get("proportional_gain", params.get("kp", 0.0))))
    ki = float(params.get("ki", params.get("integral_gain", 0.0))) if controller_type in {"pi", "pid"} else 0.0
    kd = float(params.get("kd", params.get("derivative_gain", 0.0))) if controller_type == "pid" else 0.0
    integral = float(params.get("integral_initial", 0.0))
    previous_error = float(params.get("previous_error_initial", error))
    derivative = (error - previous_error) / max(dt, 1e-12)
    raw = bias + feed_forward + kp * error + ki * integral + kd * derivative
    lower = float(params.get("lower_limit", params.get("min", -float("inf"))))
    upper = float(params.get("upper_limit", params.get("max", float("inf"))))
    command = min(max(raw, lower), upper)
    return {
        str(controller["output"]): command,
        f"controller.{name}.target": target,
        f"controller.{name}.measurement": measurement,
        f"controller.{name}.error": error,
        f"controller.{name}.command": command,
        f"controller.{name}.raw_command": raw,
        f"controller.{name}.saturated": float(command != raw),
        f"controller.{name}.integral": integral,
        f"controller.{name}.derivative": derivative,
    }
```

Update callers of `_evaluate_feedback_controller` in `controllers.py` to pass `dt` from the controller evaluation period.

- [ ] **Step 8: Run tests and verify pass**

```
pytest tests/unit/test_pid_derivative.py -v
```
Expected: PASS for all 3 tests.

- [ ] **Step 9: Commit**

```bash
git add atha/config/controllers.py atha/runner/dae_execution.py tests/unit/test_pid_derivative.py
git commit -m "fix: correct PID derivative division by dt and remove ODE previous_error state"
```

---

## Task 2: Add Simulation Phases and Phase-Based Controller Activation

**Files:**
- Modify: `atha/config/schema.py` — Add `PhaseConfig` dataclass
- Modify: `atha/runner/dae_execution.py` — Add `_current_phase()`, skip inactive controllers
- Create: `tests/unit/test_phase_control.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_phase_control.py
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock


@dataclass
class _Phase:
    name: str
    start_s: float
    end_s: float


class TestPhaseControl:
    def _make_problem(self, phases, controllers):
        from atha.runner.dae_execution import DAEExecutionProblem

        loaded = MagicMock()
        loaded.controllers = MagicMock()
        loaded.controllers.controllers = controllers
        loaded.controllers.evaluation = {"frequency_hz": 10.0}
        loaded.transients = None
        loaded.boundaries = None
        loaded.timings = None
        loaded.operating_conditions = None
        loaded.engine = MagicMock()
        loaded.engine.components = {}
        loaded.engine.connections = []

        execution_plan = MagicMock()
        execution_plan.state_modes = {}
        execution_plan.phases = phases

        problem = DAEExecutionProblem.__new__(DAEExecutionProblem)
        problem.loaded = loaded
        problem.execution_plan = execution_plan
        problem._controller_period_s = 0.1
        problem._controller_hold_cache = {}
        problem.state_names = []
        problem._controller_state_indexes = {}
        return problem

    def test_controller_inactive_outside_active_phases(self):
        """A controller with active_phases=[CLC] must produce no output during startup."""
        ctrl = {
            "type": "proportional",
            "inputs": {"target": "targets.sp", "measurement": "measurements.val"},
            "output": "valve.command",
            "active_phases": ["CLC"],
            "parameters": {"gain": 1.0, "bias": 0.5},
        }
        phases = [_Phase("startup", 0.0, 5.0), _Phase("CLC", 5.0, 20.0)]
        problem = self._make_problem(phases, {"ctrl": ctrl})

        # t=2.0 is in startup phase → controller should be inactive
        result = problem._current_phase(2.0)
        assert result == "startup"

        outputs = problem._evaluate_controllers(
            2.0, {"sp": 1.0}, {}, {"val": 0.5}, {}
        )
        # inactive controller should not set valve.command
        assert "valve.command" not in outputs

    def test_controller_active_within_active_phases(self):
        """A controller with active_phases=[CLC] must produce output during CLC."""
        ctrl = {
            "type": "proportional",
            "inputs": {"target": "targets.sp", "measurement": "measurements.val"},
            "output": "valve.command",
            "active_phases": ["CLC"],
            "parameters": {"gain": 1.0, "bias": 0.5, "lower_limit": 0.0, "upper_limit": 1.0},
        }
        phases = [_Phase("startup", 0.0, 5.0), _Phase("CLC", 5.0, 20.0)]
        problem = self._make_problem(phases, {"ctrl": ctrl})

        result = problem._current_phase(10.0)
        assert result == "CLC"

        outputs = problem._evaluate_controllers(
            10.0, {"sp": 1.0}, {}, {"val": 0.5}, {}
        )
        assert "valve.command" in outputs

    def test_controller_always_active_when_no_active_phases(self):
        """Controller without active_phases must always run."""
        ctrl = {
            "type": "proportional",
            "inputs": {"target": "targets.sp", "measurement": "measurements.val"},
            "output": "valve.command",
            "parameters": {"gain": 1.0, "bias": 0.5, "lower_limit": 0.0, "upper_limit": 1.0},
        }
        phases = [_Phase("startup", 0.0, 5.0), _Phase("CLC", 5.0, 20.0)]
        problem = self._make_problem(phases, {"ctrl": ctrl})

        for t in [1.0, 10.0]:
            outputs = problem._evaluate_controllers(t, {"sp": 1.0}, {}, {"val": 0.5}, {})
            assert "valve.command" in outputs

    def test_current_phase_returns_none_when_no_phases_defined(self):
        """If no phases are defined, _current_phase returns None."""
        problem = self._make_problem([], {})
        assert problem._current_phase(10.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_phase_control.py -v
```
Expected: FAIL — `_current_phase` method doesn't exist, `active_phases` is ignored.

- [ ] **Step 3: Add `PhaseConfig` to schema**

In `atha/config/schema.py`, add this dataclass (near the other time-related configs):

```python
@dataclass(frozen=True)
class PhaseConfig:
    """Named simulation phase with start and end times."""
    name: str
    start_s: float
    end_s: float
```

The existing analysis config loading code loads `analysis.time.phases` as a list of dicts. No schema validation change is strictly required since schema.py uses dataclasses with optional fields, but if `AnalysisTimeConfig` or equivalent exists, add:
```python
phases: list[dict] = field(default_factory=list)
```

- [ ] **Step 4: Add `_current_phase()` and phase filtering to `DAEExecutionProblem`**

In `atha/runner/dae_execution.py`, add this method to `DAEExecutionProblem`:

```python
def _current_phase(self, t: float) -> str | None:
    """Return the name of the active phase at time t, or None if no phases defined."""
    phases = getattr(self.execution_plan, "phases", [])
    if not phases:
        return None
    for phase in phases:
        start = float(getattr(phase, "start_s", getattr(phase, "start", 0.0)))
        end = float(getattr(phase, "end_s", getattr(phase, "end", float("inf"))))
        if start <= t < end:
            return str(getattr(phase, "name", ""))
    # Return last phase if t is beyond all defined end times
    last = phases[-1]
    return str(getattr(last, "name", ""))
```

- [ ] **Step 5: Add phase check to `_evaluate_controllers`**

In `_evaluate_controllers` in `dae_execution.py`, find where individual controllers are iterated. Before evaluating each controller, add:

```python
current_phase = self._current_phase(t)
for name, single in ordered_controllers:
    active_phases = single.get("active_phases")
    if active_phases is not None and current_phase is not None:
        if current_phase not in active_phases:
            continue  # controller inactive in this phase
    # ... existing evaluation code ...
```

- [ ] **Step 6: Surface phases from execution plan**

In `atha/runner/solver_driver.py`, ensure `ExecutionPlan` carries a `phases` field. Add if missing:
```python
@dataclass(frozen=True)
class ExecutionPlan:
    # existing fields...
    phases: list = field(default_factory=list)  # list of PhaseConfig
```

And in `build_execution_plan()`, parse phases from `loaded.analysis` config:
```python
raw_phases = getattr(getattr(loaded, "analysis", None), "time", {})
if isinstance(raw_phases, dict):
    raw_phases = raw_phases.get("phases", [])
elif hasattr(raw_phases, "phases"):
    raw_phases = raw_phases.phases
else:
    raw_phases = []

from atha.config.schema import PhaseConfig
phases = [PhaseConfig(name=p["name"], start_s=float(p["start_s"]), end_s=float(p["end_s"])) for p in raw_phases if isinstance(p, dict)]
```

- [ ] **Step 7: Run tests and verify pass**

```
pytest tests/unit/test_phase_control.py -v
```
Expected: PASS for all 4 tests.

- [ ] **Step 8: Commit**

```bash
git add atha/config/schema.py atha/runner/dae_execution.py atha/runner/solver_driver.py tests/unit/test_phase_control.py
git commit -m "feat: add simulation phase-based controller activation"
```

---

## Task 3: Fix Pump Map to Use φ/ψ Coefficients

**Files:**
- Modify: `atha/components/residuals.py` — `PumpHeadContract.evaluate()`
- Modify: `atha/components/pump.py` — `compute_outputs()`
- Create: `tests/unit/test_pump_map.py`

**Physics:**
```
φ = ṁ / (ρ_design × N × D³)        [dimensionless flow coefficient]
ψ = ΔP / (ρ_design × N² × D²)       [dimensionless head coefficient]
η = efficiency from map at φ

ΔP_target = ρ_design × ψ(φ) × N² × D²
h_out = h_in + ΔP / (ρ × η)
```

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pump_map.py
import pytest
from unittest.mock import MagicMock
from atha.components.residuals import PumpHeadContract, ResidualEvaluationContext


def _make_component(name="lox_pump", diameter=0.145, rho_design=1140.0,
                    mdot_design=30.5, dP_design=13.0e6, speed_design=32000.0,
                    efficiency_design=0.74):
    comp = MagicMock()
    comp.name = name
    comp.parameters = {
        "diameter": diameter,
        "pump_map": {
            "rho_design": rho_design,
            "mdot_design": mdot_design,
            "dP_design": dP_design,
            "speed_design": speed_design,
            "efficiency_design": efficiency_design,
        }
    }
    comp.maps = {}
    return comp


def _make_context(z: dict, inputs: dict = None, model: dict = None):
    return ResidualEvaluationContext(z=z, inputs=inputs or {}, model=model or {})


class TestPumpHeadContract:
    def test_design_point_residual_zero(self):
        """At design speed and design flow, the residual must be zero."""
        import math
        D = 0.145
        rho = 1140.0
        N_rpm = 32000.0
        N = N_rpm * 2 * math.pi / 60.0   # rad/s
        mdot = 30.5
        dP_design = 13.0e6

        phi = mdot / (rho * N * D**3)
        psi = dP_design / (rho * N**2 * D**2)

        # Mock efficiency map returning design efficiency at phi
        eff_map = MagicMock()
        eff_map.evaluate.return_value = {"eta": 0.74}

        # Mock head map returning design psi at phi
        head_map = MagicMock()
        head_map.evaluate.return_value = {"psi": psi}

        model = {
            "lox_pump.map.head_map": head_map,
            "lox_pump.map.head_map.output": "psi",
            "lox_pump.map.efficiency_map": eff_map,
            "lox_pump.map.efficiency_map.output": "eta",
        }

        z = {"lox_pump.delta_P": dP_design}
        inputs = {
            "lox_pump.shaft.omega": N,
            "lox_pump.mdot": mdot,
            "lox_pump.inlet.rho": rho,
        }
        context = _make_context(z, inputs, model)
        comp = _make_component()

        contract = PumpHeadContract()
        result = contract.evaluate(comp, context)

        assert abs(result["lox_pump.delta_P_residual"]) < 1.0, (
            f"Residual at design point should be ~0, got {result['lox_pump.delta_P_residual']}"
        )

    def test_returns_efficiency_for_downstream_use(self):
        """PumpHeadContract must return efficiency so pump.py can compute h_out."""
        import math
        D = 0.145
        rho = 1140.0
        N = 32000.0 * 2 * math.pi / 60.0
        mdot = 30.5
        dP_design = 13.0e6
        psi = dP_design / (rho * N**2 * D**2)

        eff_map = MagicMock()
        eff_map.evaluate.return_value = {"eta": 0.74}
        head_map = MagicMock()
        head_map.evaluate.return_value = {"psi": psi}

        model = {
            "lox_pump.map.head_map": head_map,
            "lox_pump.map.head_map.output": "psi",
            "lox_pump.map.efficiency_map": eff_map,
            "lox_pump.map.efficiency_map.output": "eta",
        }
        z = {"lox_pump.delta_P": dP_design}
        inputs = {"lox_pump.shaft.omega": N, "lox_pump.mdot": mdot, "lox_pump.inlet.rho": rho}
        context = _make_context(z, inputs, model)
        comp = _make_component()

        result = PumpHeadContract().evaluate(comp, context)
        assert "lox_pump.efficiency" in result, "Contract must return efficiency for h_out calculation"

    def test_fallback_to_affinity_when_no_map(self):
        """Without a head_map, must fall back to affinity-law scaling."""
        import math
        D = 0.145
        rho = 1140.0
        N_design = 32000.0 * 2 * math.pi / 60.0
        N_actual = N_design * 0.8  # 80% speed
        mdot = 30.5
        dP_design = 13.0e6

        z = {"lox_pump.delta_P": dP_design * 0.64}  # affinity law: 0.8² = 0.64
        inputs = {"lox_pump.shaft.omega": N_actual, "lox_pump.mdot": mdot, "lox_pump.inlet.rho": rho}
        context = _make_context(z, inputs, {})
        comp = _make_component()

        result = PumpHeadContract().evaluate(comp, context)
        # Residual should be near 0 because delta_P matches affinity law prediction
        assert abs(result["lox_pump.delta_P_residual"]) < dP_design * 0.02


class TestPumpEnthalpyRise:
    def test_outlet_enthalpy_greater_than_inlet(self):
        """Pump work must raise fluid enthalpy: h_out > h_in."""
        from atha.components.pump import Pump, PumpMap

        fluid = MagicMock()
        fluid_state = MagicMock()
        fluid_state.rho = 1140.0
        fluid_state.T = 90.0
        fluid.state_from_Ph.return_value = fluid_state

        pump_map = PumpMap(mdot_design=30.5, dP_design=13.0e6, omega_design=3351.0, eta_design=0.74)
        pump = Pump("p", diameter=0.145, pump_map=pump_map, fluid=fluid)

        import math
        N = 32000.0 * 2 * math.pi / 60.0
        outputs = pump.compute_outputs(0.0, {}, {
            "shaft.omega": N,
            "inlet.P": 1.0e6,
            "inlet.h": 0.0,
            "inlet.mdot": 30.5,
        })

        assert outputs["outlet.h"] > 0.0, (
            f"Outlet enthalpy must exceed inlet (0.0), got {outputs['outlet.h']}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/unit/test_pump_map.py -v
```
Expected: FAIL — `PumpHeadContract` ignores efficiency map, `compute_outputs` returns `h_out = h_in`.

- [ ] **Step 3: Rewrite `PumpHeadContract.evaluate()` in `residuals.py`**

Replace the existing `evaluate` method (lines 252-280) with:

```python
def evaluate(self, component: ComponentConfig, context: ResidualEvaluationContext) -> dict[str, float]:
    name = component.name
    omega = context.value(f"{name}.shaft.omega", context.value(f"{name}.omega", _design_omega(component, 1.0)))
    omega_safe = max(abs(omega), 1.0e-12)

    mdot = abs(
        context.value(
            f"{name}.mdot",
            context.value(f"{name}.inlet.mdot", context.value(f"{name}.outlet.mdot",
                _nested_param(component, "pump_map", "mdot_design", 1.0))),
        )
    )

    D = float(component.parameters.get("diameter", 0.0))
    rho_design = float(_nested_param(component, "pump_map", "rho_design",
        context.value(f"{name}.inlet.rho", 1000.0)))
    rho_design = max(rho_design, 1.0)

    # --- φ/ψ path (preferred) ---
    if D > 0.0:
        phi = mdot / max(rho_design * omega_safe * D**3, 1.0e-30)

        # Look up head coefficient ψ from map
        head_values = _evaluate_map(context, component, "head_map", {"phi": phi})
        psi = _first_map_value(head_values, ("psi", "head_coefficient"))

        if psi is not None:
            d_p_target = rho_design * psi * omega_safe**2 * D**2
        else:
            # Head map exists but does not have psi column; fall back
            dP_via_pressure = _first_map_value(head_values, ("pressure_rise", "head", "delta_P"))
            if dP_via_pressure is not None:
                d_p_target = dP_via_pressure
            else:
                # No map at all — affinity law
                omega_design = max(_design_omega(component, omega_safe), 1.0e-12)
                d_p_design = float(_nested_param(component, "pump_map", "dP_design",
                    component.parameters.get("delta_P_design", 0.0)))
                d_p_target = d_p_design * (omega_safe / omega_design) ** 2

        # Look up efficiency from efficiency_map at same φ
        eff_values = _evaluate_map(context, component, "efficiency_map", {"phi": phi})
        eta = _first_map_value(eff_values, ("eta", "efficiency"))
        if eta is None:
            eta = float(_nested_param(component, "pump_map", "efficiency_design",
                component.parameters.get("efficiency_design", 0.74)))
        eta = min(max(eta, 1.0e-6), 1.0)

    else:
        # --- Backward-compat path: speed_ratio/flow_ratio map ---
        omega_design = max(_design_omega(component, omega_safe), 1.0e-12)
        mdot_design = max(float(_nested_param(component, "pump_map", "mdot_design",
            component.parameters.get("mdot_design", mdot if mdot else 1.0))), 1.0e-12)
        speed_ratio = omega_safe / omega_design
        flow_ratio = mdot / mdot_design
        head_values = _evaluate_map(context, component, "head_map", {
            "speed_ratio": speed_ratio, "flow_ratio": flow_ratio,
            "corrected_speed": speed_ratio, "corrected_flow": flow_ratio,
        })
        d_p_target = _first_map_value(head_values, ("pressure_rise", "head", "delta_P"))
        if d_p_target is None:
            d_p_design = float(_nested_param(component, "pump_map", "dP_design",
                component.parameters.get("delta_P_design", 0.0)))
            d_p_target = d_p_design * speed_ratio**2

        eff_values = _evaluate_map(context, component, "efficiency_map", {
            "speed_ratio": speed_ratio, "flow_ratio": flow_ratio,
        })
        eta = _first_map_value(eff_values, ("eta", "efficiency"))
        if eta is None:
            eta = float(_nested_param(component, "pump_map", "efficiency_design", 0.74))
        eta = min(max(eta, 1.0e-6), 1.0)

    return {
        f"{name}.delta_P_residual": context.value(f"{name}.delta_P") - d_p_target,
        f"{name}.efficiency": eta,
    }
```

- [ ] **Step 4: Fix `compute_outputs()` in `pump.py` to compute enthalpy rise**

Replace lines 205-222 (the `return` block) with:

```python
        # Enthalpy rise: h_out = h_in + ΔP / (ρ × η)
        delta_h = delta_P / (rho * max(eta, 1e-6))
        h_out = h_in + delta_h

        # Outlet temperature from fluid model
        try:
            _fs_out = self._fluid.state_from_Ph(P_out, h_out)
            T_out = _fs_out.T
        except Exception:
            T_out = T_in + delta_h / 4186.0  # fallback: approximate Cp for water

        return {
            "outlet.P":   P_out,
            "outlet.h":   h_out,
            "outlet.T":   T_out,
            "outlet.rho": rho,
            "P":    P_out,
            "h":    h_out,
            "T":    T_out,
            "rho":  rho,
            "delta_P":    delta_P,
            "inlet.mdot": mdot,
            "mdot":       mdot,
            "power":      W,
            "tau_load":   tau,
            "efficiency": eta,
        }
```

- [ ] **Step 5: Run pump map tests**

```
pytest tests/unit/test_pump_map.py -v
```
Expected: PASS for all tests.

- [ ] **Step 6: Commit**

```bash
git add atha/components/residuals.py atha/components/pump.py tests/unit/test_pump_map.py
git commit -m "feat: implement phi/psi pump map with efficiency-based enthalpy rise"
```

---

## Task 4: Update Example 19 Pump Maps to φ/ψ Format

**Files:**
- Modify: `examples/19_ffsc_dae_acceptance/configs/maps/lox_pump_affinity.yaml`
- Modify: `examples/19_ffsc_dae_acceptance/configs/maps/lox_pump_affinity.csv`
- Modify: `examples/19_ffsc_dae_acceptance/configs/maps/methane_pump_affinity.yaml`
- Modify: `examples/19_ffsc_dae_acceptance/configs/maps/methane_pump_affinity.csv`
- Modify: `examples/19_ffsc_dae_acceptance/configs/engine.yaml` — add `rho_design`, verify `diameter`

- [ ] **Step 1: Update `lox_pump_affinity.yaml`**

```yaml
name: lox_pump_affinity
kind: structured_grid
source:
  type: csv
  path: lox_pump_affinity.csv
axes:
  - name: phi
    column: phi
outputs:
  - name: psi
    column: psi
  - name: eta
    column: eta
interpolation:
  method: linear
```

- [ ] **Step 2: Update `lox_pump_affinity.csv`**

Design point: φ_design=0.002619, ψ_design=0.04831, η_design=0.74. Map spans ±50% of design φ with realistic efficiency bell curve:

```
phi,psi,eta
0.001000,0.04831,0.55
0.001500,0.04831,0.64
0.002000,0.04831,0.70
0.002619,0.04831,0.74
0.003200,0.04831,0.71
0.004000,0.04831,0.66
0.005000,0.04831,0.57
```

Note: ψ is constant (affinity law — head coefficient is speed-independent). η peaks at design φ.

- [ ] **Step 3: Update `methane_pump_affinity.yaml`**

```yaml
name: methane_pump_affinity
kind: structured_grid
source:
  type: csv
  path: methane_pump_affinity.csv
axes:
  - name: phi
    column: phi
outputs:
  - name: psi
    column: psi
  - name: eta
    column: eta
interpolation:
  method: linear
```

- [ ] **Step 4: Update `methane_pump_affinity.csv`**

Design point: φ_design=0.006879, ψ_design=0.36292, η_design=0.69:

```
phi,psi,eta
0.002500,0.36292,0.50
0.004000,0.36292,0.60
0.005500,0.36292,0.66
0.006879,0.36292,0.69
0.009000,0.36292,0.65
0.012000,0.36292,0.57
0.015000,0.36292,0.48
```

- [ ] **Step 5: Update `engine.yaml` pump parameters**

In `examples/19_ffsc_dae_acceptance/configs/engine.yaml`, add `rho_design` and `diameter` to each pump's `pump_map`:

```yaml
lox_pump:
  type: Pump
  parameters:
    diameter: 0.145
    pump_map:
      model: affinity_law
      mdot_design: 30.5
      dP_design: 13.0e6
      speed_design: 32000
      efficiency_design: 0.74
      rho_design: 1140.0      # LOX density at operating conditions [kg/m³]
  maps:
    head_map:
      ref: lox_pump_affinity
      output: psi
    efficiency_map:
      ref: lox_pump_affinity
      output: eta

methane_pump:
  type: Pump
  parameters:
    diameter: 0.105
    pump_map:
      model: affinity_law
      mdot_design: 9.5
      dP_design: 13.5e6
      speed_design: 27000
      efficiency_design: 0.69
      rho_design: 422.0       # Liquid methane density [kg/m³]
  maps:
    head_map:
      ref: methane_pump_affinity
      output: psi
    efficiency_map:
      ref: methane_pump_affinity
      output: eta
```

- [ ] **Step 6: Commit**

```bash
git add examples/19_ffsc_dae_acceptance/configs/maps/ examples/19_ffsc_dae_acceptance/configs/engine.yaml
git commit -m "feat: convert Example 19 pump maps to phi/psi dimensionless coefficients"
```

---

## Task 5: Update Example 19 for Phase-Based Controller Activation

**Files:**
- Modify: `examples/19_ffsc_dae_acceptance/configs/analysis.yaml`
- Modify: `examples/19_ffsc_dae_acceptance/configs/controller.yaml`

- [ ] **Step 1: Add `phases` to `analysis.yaml`**

In `analysis.yaml`, under `analysis.time`, add:

```yaml
analysis:
  type: ffsc_dae_transient
  time:
    start_s: 0.0
    end_s: 25.0
    phases:
      - name: startup
        start_s: 0.0
        end_s: 3.0
      - name: CLC
        start_s: 3.0
        end_s: 22.0
      - name: shutdown
        start_s: 22.0
        end_s: 25.0
```

- [ ] **Step 2: Add `active_phases` to `controller.yaml`**

```yaml
name: ffsc_dae_acceptance_controller
evaluation:
  frequency_hz: 20.0
controllers:
  methane_crossover_mdot_p:
    type: proportional
    active_phases: [CLC]
    inputs:
      target: targets.mdot_total
      measurement: measurements.mdot_total
    output: methane_crossover_valve.command
    parameters:
      bias: 0.5
      gain: 0.08
      lower_limit: 0.05
      upper_limit: 1.00
  lox_crossover_of_p:
    type: proportional
    active_phases: [CLC]
    inputs:
      target: targets.OF
      measurement: measurements.OF
    output: lox_crossover_valve.command
    parameters:
      bias: 0.50
      gain: -5
      lower_limit: 0.05
      upper_limit: 1.00
```

- [ ] **Step 3: Commit**

```bash
git add examples/19_ffsc_dae_acceptance/configs/analysis.yaml examples/19_ffsc_dae_acceptance/configs/controller.yaml
git commit -m "config: add simulation phases and active_phases to Example 19 controllers"
```

---

## Task 6: Update Example 20 Pump Maps and Phase Controllers

**Files:**
- Modify: `examples/20_gg_single_shaft_methalox/configs/maps/lox_pump_affinity.yaml`
- Modify: `examples/20_gg_single_shaft_methalox/configs/maps/methane_pump_affinity.yaml`
- Modify: `examples/20_gg_single_shaft_methalox/configs/engine.yaml`
- Modify: `examples/20_gg_single_shaft_methalox/configs/analysis.yaml`
- Modify: `examples/20_gg_single_shaft_methalox/configs/controller.yaml`

- [ ] **Step 1: Update `lox_pump_affinity.yaml` (Example 20)**

Example 20 currently uses an analytic constant map. Switch to constant `psi`:

```yaml
name: lox_pump_affinity
kind: analytic
source:
  type: constant
  values:
    psi: 0.04460
    eta: 0.74
```

- [ ] **Step 2: Update `methane_pump_affinity.yaml` (Example 20)**

```yaml
name: methane_pump_affinity
kind: analytic
source:
  type: constant
  values:
    psi: 0.32593
    eta: 0.69
```

- [ ] **Step 3: Update `engine.yaml` (Example 20)**

Add `rho_design`, `diameter`, and update map outputs:

```yaml
lox_pump:
  type: Pump
  parameters:
    diameter: 0.145
    pump_map:
      mdot_design: 30.48
      dP_design: 12.0e6
      speed_design: 32000
      efficiency_design: 0.74
      rho_design: 1140.0
  maps:
    head_map: {ref: lox_pump_affinity, output: psi}
    efficiency_map: {ref: lox_pump_affinity, output: eta}

methane_pump:
  type: Pump
  parameters:
    diameter: 0.090
    pump_map:
      mdot_design: 9.52
      dP_design: 12.5e6
      speed_design: 32000
      efficiency_design: 0.69
      rho_design: 422.0
  maps:
    head_map: {ref: methane_pump_affinity, output: psi}
    efficiency_map: {ref: methane_pump_affinity, output: eta}
```

- [ ] **Step 4: Update `analysis.yaml` (Example 20)**

```yaml
analysis:
  type: gg_single_shaft_transient
  time:
    start_s: 0.0
    end_s: 25.01
    phases:
      - name: startup
        start_s: 0.0
        end_s: 3.0
      - name: CLC
        start_s: 3.0
        end_s: 22.0
      - name: shutdown
        start_s: 22.0
        end_s: 25.01
```

- [ ] **Step 5: Update `controller.yaml` (Example 20)**

```yaml
name: gg_single_shaft_methalox_controller
evaluation:
  frequency_hz: 10.0
controllers:
  lox_generator_mdot_p:
    type: pid
    active_phases: [CLC]
    inputs:
      target: targets.mdot_total
      measurement: measurements.mdot_total
    output: lox_generator_valve.command
    parameters:
      bias: 0.50
      proportional_gain: 0.12
      derivative_gain: 4
      lower_limit: 0.05
      upper_limit: 1.00
  methane_generator_of_p:
    type: pid
    active_phases: [CLC]
    inputs:
      target: targets.OF
      measurement: measurements.OF
    output: methane_generator_valve.command
    parameters:
      bias: 2
      proportional_gain: -4
      derivative_gain: 1
      lower_limit: 0.05
      upper_limit: 1.00
```

- [ ] **Step 6: Commit**

```bash
git add examples/20_gg_single_shaft_methalox/configs/
git commit -m "config: update Example 20 pump maps to psi coefficients and add phase control"
```

---

## Task 7: Verify Both Examples Run End-to-End

This task ensures Examples 19 and 20 execute without error and produce physically reasonable output.

- [ ] **Step 1: Run Example 19**

From the repo root:
```bash
python -m atha.runner examples/19_ffsc_dae_acceptance/configs/analysis.yaml
```
Expected: Run completes, CSV written, no solver crash.

- [ ] **Step 2: Verify Example 19 physical behavior**

Inspect the output CSV to confirm:
- `lox_pump.efficiency` is in the range [0.5, 0.85] (not 0)
- `lox_pump.outlet.h > lox_pump.inlet.h` (enthalpy rises)
- During CLC phase (t=3-22s), `controller.methane_crossover_mdot_p.derivative` is non-zero when error changes
- During startup (t<3s), controllers produce no command output

- [ ] **Step 3: Run Example 20**

```bash
python -m atha.runner examples/20_gg_single_shaft_methalox/configs/analysis.yaml
```
Expected: Run completes. For `gg_single_shaft_transient` analysis type, this may invoke `run_gg_single_shaft_transient`.

- [ ] **Step 4: Verify Example 20 physical behavior**

Inspect output CSV:
- PID derivative is non-zero for `lox_generator_mdot_p` and `methane_generator_of_p` during CLC
- Pump efficiency in reasonable range
- Outlet enthalpy higher than inlet

- [ ] **Step 5: Run existing unit tests**

```bash
pytest tests/unit/ -v
```
Expected: All tests pass.

- [ ] **Step 6: Run integration regression tests if available**

```bash
pytest tests/integration/ -v -m "not slow"
```
Expected: Pass or skip.

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "feat: complete pump phi/psi, PID derivative fix, and phase-based controller activation"
```

---

## Verification Checklist

| Feature | How to verify |
|---|---|
| φ/ψ pump map used | Check CSV output: `lox_pump.efficiency` present and non-trivial |
| Outlet enthalpy rises | `lox_pump.outlet.h > lox_pump.inlet.h` in CSV |
| PID derivative non-zero | `controller.lox_generator_mdot_p.derivative` varies in Example 20 CSV |
| Phase activation | Example 19: no controller output at t=1s; controller active at t=10s |
| Example 19 runs | `python -m atha.runner examples/19_ffsc_dae_acceptance/configs/analysis.yaml` exits 0 |
| Example 20 runs | `python -m atha.runner examples/20_gg_single_shaft_methalox/configs/analysis.yaml` exits 0 |
| Unit tests pass | `pytest tests/unit/ -v` all green |
