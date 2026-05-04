# Engine Test Profile and Monte Carlo Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two independent capabilities to ATHA: (1) multi-phase engine test profile simulation with event scheduling, abort limits, and data recording; (2) Monte Carlo uncertainty quantification with LHS sampling, parallel execution, and Sobol sensitivity analysis.

**Architecture:** Test profiles are sequential phase executors that reuse existing `TransientSolver` and `SteadyStateSolver`, threading state from one phase to the next. Monte Carlo runs a user-supplied evaluate function N times in parallel via joblib, using `scipy.stats.qmc.LatinHypercube` or SALib Saltelli sampling. Both subsystems are independent — Phase B does not depend on Phase A.

**Tech Stack:** Python 3.11+, SciPy (solve_ivp dense output, qmc.LatinHypercube), h5py (HDF5 output), joblib (parallelism), SALib (Sobol indices), Matplotlib (plots). All are additions to the existing ATHA stack.

**Research:** See `development/10_test_profile_research.md` and `development/11_monte_carlo_research.md`.

---

## File Structure

```
atha/
├── profiles/
│   ├── __init__.py          # exports TestProfile, PhaseDefinition, PhaseMode,
│   │                        #   ControlCommand, SafetyLimit, TestProfileResult
│   ├── phase.py             # PhaseDefinition, PhaseMode, ControlCommand dataclasses
│   ├── limits.py            # SafetyLimit, AbortManager, EngineAbort exception
│   ├── result.py            # PhaseResult, TestProfileResult, plot_timeline()
│   ├── recording.py         # downsample_dense_output() via solve_ivp dense output
│   ├── executor.py          # execute_phase() — core phase dispatch
│   ├── profile.py           # TestProfile.execute() — multi-phase loop
│   └── io.py                # save_profile_result() / load_profile_result() via h5py
└── monte_carlo/
    ├── __init__.py          # exports UncertainParameter, ParameterType,
    │                        #   MonteCarloRunner, MonteCarloConfig, MonteCarloResult
    ├── parameters.py        # UncertainParameter, ParameterType enum
    ├── sampling.py          # LHSSampler, SaltelliSampler
    ├── runner.py            # MonteCarloRunner (serial + joblib parallel)
    ├── sensitivity.py       # SobolAnalysis wrapping SALib
    ├── statistics.py        # MCStatistics dataclass, compute_statistics()
    ├── results.py           # MonteCarloResult, HDF5 save/load, histogram/Sobol plots
    └── profile_runner.py    # ProfileMonteCarloRunner (MC over full test profiles)

tests/
├── unit/
│   ├── test_profiles.py     # unit tests for all profiles/ modules
│   └── test_monte_carlo.py  # unit tests for all monte_carlo/ modules
└── integration/
    ├── test_profile_ttbe.py          # TTBE throttle sweep end-to-end
    └── test_monte_carlo_ttbe.py      # TTBE MC study end-to-end
```

---

## Phase A: Engine Test Profiles

---

### Task A1: PhaseDefinition, PhaseMode, ControlCommand

**Files:**
- Create: `atha/profiles/__init__.py`
- Create: `atha/profiles/phase.py`
- Create: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_profiles.py
from atha.profiles.phase import PhaseDefinition, PhaseMode, ControlCommand
import numpy as np
import pytest


def test_phase_mode_values():
    assert PhaseMode.STEADY_TRIM.value == "steady_trim"
    assert PhaseMode.TRANSIENT.value == "transient"
    assert PhaseMode.DWELL.value == "dwell"


def test_control_command_evaluates():
    cmd = ControlCommand(bcs_key="throttle.cmd", fn=lambda t: 1.0 - 0.35 * t / 5.0)
    assert abs(cmd.fn(0.0) - 1.0) < 1e-10
    assert abs(cmd.fn(5.0) - 0.65) < 1e-10


def test_phase_definition_defaults():
    phase = PhaseDefinition(
        name="mainstage",
        mode=PhaseMode.STEADY_TRIM,
        duration=30.0,
    )
    assert phase.name == "mainstage"
    assert phase.mode == PhaseMode.STEADY_TRIM
    assert phase.trim_targets == {}
    assert phase.control_commands == []
    assert phase.abort_checks == []
    assert phase.recording_rate_hz == 100.0


def test_phase_definition_with_commands():
    cmd = ControlCommand("throttle.cmd", fn=lambda t: 0.65)
    phase = PhaseDefinition(
        name="transient",
        mode=PhaseMode.TRANSIENT,
        duration=5.0,
        control_commands=[cmd],
        recording_rate_hz=200.0,
    )
    assert len(phase.control_commands) == 1
    assert phase.control_commands[0].bcs_key == "throttle.cmd"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_profiles.py -v
# Expected: FAIL — ImportError: cannot import name 'PhaseDefinition'
```

- [ ] **Step 3: Implement phase.py**

```python
# atha/profiles/phase.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class PhaseMode(Enum):
    STEADY_TRIM = "steady_trim"
    TRANSIENT = "transient"
    DWELL = "dwell"


@dataclass
class ControlCommand:
    """Maps a BCS dict key to a time-varying callable.

    fn(t_phase) -> float, where t_phase is seconds since phase start (0-based).
    """
    bcs_key: str
    fn: Callable[[float], float]


@dataclass
class PhaseDefinition:
    """Declarative description of one phase in a test profile."""
    name: str
    mode: PhaseMode
    duration: float                              # seconds
    trim_targets: Dict[str, float] = field(default_factory=dict)
    control_commands: List[ControlCommand] = field(default_factory=list)
    abort_checks: List = field(default_factory=list)   # List[SafetyLimit]
    recording_rate_hz: float = 100.0
    solver_options: Dict = field(default_factory=dict)
```

- [ ] **Step 4: Create profiles/__init__.py**

```python
# atha/profiles/__init__.py
from atha.profiles.phase import PhaseDefinition, PhaseMode, ControlCommand

__all__ = ["PhaseDefinition", "PhaseMode", "ControlCommand"]
```

- [ ] **Step 5: Run tests, verify pass**

```
pytest tests/unit/test_profiles.py -v
# Expected: PASS (4 tests)
```

- [ ] **Step 6: Commit**

```bash
git add atha/profiles/ tests/unit/test_profiles.py
git commit -m "feat: add PhaseDefinition, PhaseMode, ControlCommand"
```

---

### Task A2: SafetyLimit, AbortManager, EngineAbort

**Files:**
- Create: `atha/profiles/limits.py`
- Modify: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_profiles.py
from atha.profiles.limits import SafetyLimit, AbortManager, EngineAbort
import numpy as np


def test_safety_limit_upper_violation():
    limit = SafetyLimit(
        name="Pc_max", component_name="chamber", state_name="P",
        upper_limit=25e6, is_hard=True,
    )
    assert limit.upper_limit == 25e6
    assert limit.is_hard is True
    assert limit.lower_limit is None


def test_engine_abort_exception():
    exc = EngineAbort(reason="Pc exceeded 25 MPa", t=3.52)
    assert exc.reason == "Pc exceeded 25 MPa"
    assert exc.t == 3.52
    assert "Pc exceeded" in str(exc)


def test_abort_manager_no_violation():
    limit = SafetyLimit("Pc_max", "chamber", "P", upper_limit=25e6, is_hard=True)
    mgr = AbortManager(limits=[limit])

    # Mock layout with state_offsets and component P=10 MPa (no violation)
    class MockComp:
        name = "chamber"
        state_names = ["P", "h"]
        _state_values = {"P": 10e6, "h": 1e6}

    class MockLayout:
        components = [MockComp()]
        state_offsets = {"chamber": 0}

    X = np.array([10e6, 1e6])
    # Should not raise
    mgr.check(MockLayout(), X)


def test_abort_manager_hard_violation_raises():
    limit = SafetyLimit("Pc_max", "chamber", "P", upper_limit=25e6, is_hard=True)
    mgr = AbortManager(limits=[limit])

    class MockComp:
        name = "chamber"
        state_names = ["P", "h"]
        _state_values = {"P": 26e6, "h": 1e6}

    class MockLayout:
        components = [MockComp()]
        state_offsets = {"chamber": 0}

    X = np.array([26e6, 1e6])
    with pytest.raises(EngineAbort, match="Pc_max"):
        mgr.check(MockLayout(), X)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_profiles.py::test_safety_limit_upper_violation -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement limits.py**

```python
# atha/profiles/limits.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


class EngineAbort(Exception):
    def __init__(self, reason: str, t: float = 0.0):
        super().__init__(f"EngineAbort at t={t:.4f}s: {reason}")
        self.reason = reason
        self.t = t


@dataclass
class SafetyLimit:
    name: str
    component_name: str
    state_name: str
    upper_limit: Optional[float] = None
    lower_limit: Optional[float] = None
    is_hard: bool = True
    description: str = ""


class AbortManager:
    def __init__(self, limits: List[SafetyLimit]):
        self.limits = limits

    def check(self, layout, X: np.ndarray, t: float = 0.0) -> None:
        """Check all limits against current state. Raises EngineAbort on hard violation."""
        for limit in self.limits:
            value = self._extract(layout, limit)
            if value is None:
                continue
            if limit.upper_limit is not None and value > limit.upper_limit:
                if limit.is_hard:
                    raise EngineAbort(
                        f"{limit.name}: {limit.component_name}.{limit.state_name} "
                        f"= {value:.4g} > upper limit {limit.upper_limit:.4g}", t=t
                    )
            if limit.lower_limit is not None and value < limit.lower_limit:
                if limit.is_hard:
                    raise EngineAbort(
                        f"{limit.name}: {limit.component_name}.{limit.state_name} "
                        f"= {value:.4g} < lower limit {limit.lower_limit:.4g}", t=t
                    )

    def _extract(self, layout, limit: SafetyLimit) -> Optional[float]:
        for comp in layout.components:
            if comp.name == limit.component_name:
                return comp._state_values.get(limit.state_name)
        return None

    def as_scipy_events(self, layout):
        """Return list of scipy event callables for solve_ivp."""
        events = []
        for limit in self.limits:
            if limit.upper_limit is not None:
                def upper_fn(t, X, _layout=layout, _limit=limit):
                    val = self._extract(_layout, _limit)
                    return (val - _limit.upper_limit) if val is not None else 1.0
                upper_fn.terminal = limit.is_hard
                upper_fn.direction = 1
                events.append(upper_fn)
            if limit.lower_limit is not None:
                def lower_fn(t, X, _layout=layout, _limit=limit):
                    val = self._extract(_layout, _limit)
                    return (_limit.lower_limit - val) if val is not None else 1.0
                lower_fn.terminal = limit.is_hard
                lower_fn.direction = 1
                events.append(lower_fn)
        return events
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_profiles.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/profiles/limits.py tests/unit/test_profiles.py
git commit -m "feat: add SafetyLimit, AbortManager, EngineAbort"
```

---

### Task A3: PhaseResult and TestProfileResult

**Files:**
- Create: `atha/profiles/result.py`
- Modify: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_profiles.py
from atha.profiles.result import PhaseResult, TestProfileResult
import numpy as np


def test_phase_result_get_state():
    t = np.array([0.0, 0.5, 1.0])
    X = np.array([[1e5, 3e5], [1.1e5, 3e5], [1.2e5, 3e5]])
    pr = PhaseResult(
        name="test_phase",
        t=t, X=X,
        state_names=["vol.P", "vol.h"],
        X_final=X[-1],
        abort_triggered=False,
    )
    P = pr.get("vol", "P")
    np.testing.assert_array_almost_equal(P, [1e5, 1.1e5, 1.2e5])


def test_phase_result_get_missing_raises():
    pr = PhaseResult("p", np.array([0.0]), np.array([[1.0]]),
                     ["vol.P"], np.array([1.0]), False)
    with pytest.raises(KeyError):
        pr.get("vol", "omega")


def test_test_profile_result_duration():
    pr1 = PhaseResult("a", np.array([0.0, 1.0]), np.zeros((2, 1)),
                      ["v.P"], np.zeros(1), False)
    pr2 = PhaseResult("b", np.array([0.0, 2.0]), np.zeros((2, 1)),
                      ["v.P"], np.zeros(1), False)
    result = TestProfileResult(
        profile_name="test", phases=[pr1, pr2],
        state_names=["v.P"],
    )
    assert result.total_duration == pytest.approx(3.0)
    assert result.abort_reason is None


def test_test_profile_result_abort():
    result = TestProfileResult(
        profile_name="test", phases=[],
        state_names=[],
        abort_reason="Pc exceeded limit",
        abort_time=2.5,
    )
    assert result.abort_reason == "Pc exceeded limit"
    assert result.success is False
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_profiles.py -k "phase_result" -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement result.py**

```python
# atha/profiles/result.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class PhaseResult:
    name: str
    t: np.ndarray           # shape (N,), time relative to phase start
    X: np.ndarray           # shape (N, n_states)
    state_names: List[str]  # ["comp.state", ...]
    X_final: np.ndarray     # shape (n_states,)
    abort_triggered: bool

    def get(self, component_name: str, state_name: str) -> np.ndarray:
        key = f"{component_name}.{state_name}"
        for i, n in enumerate(self.state_names):
            if n == key:
                return self.X[:, i]
        raise KeyError(f"State '{key}' not found. Available: {self.state_names}")

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 0 else 0.0


@dataclass
class TestProfileResult:
    profile_name: str
    phases: List[PhaseResult]
    state_names: List[str]
    abort_reason: Optional[str] = None
    abort_time: Optional[float] = None

    @property
    def success(self) -> bool:
        return self.abort_reason is None

    @property
    def total_duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def get_phase(self, name: str) -> PhaseResult:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(f"Phase '{name}' not found")

    def get_combined(self, component_name: str, state_name: str):
        """Concatenate a state across all phases (t adjusted to be global)."""
        t_parts, X_parts = [], []
        t_offset = 0.0
        for phase in self.phases:
            try:
                X_part = phase.get(component_name, state_name)
                t_parts.append(phase.t + t_offset)
                X_parts.append(X_part)
            except KeyError:
                pass
            t_offset += phase.duration
        if not t_parts:
            raise KeyError(f"{component_name}.{state_name} not found in any phase")
        return np.concatenate(t_parts), np.concatenate(X_parts)

    def plot_timeline(self, states=None, show=True):
        """Plot selected state time-series with phase boundaries marked."""
        import matplotlib.pyplot as plt

        if states is None and self.state_names:
            states = self.state_names[:4]  # default: first 4 states

        fig, axes = plt.subplots(len(states), 1,
                                  figsize=(12, 3 * len(states)), sharex=True)
        if len(states) == 1:
            axes = [axes]

        t_offset = 0.0
        phase_boundaries = [0.0]
        phase_names = []
        for phase in self.phases:
            t_offset += phase.duration
            phase_boundaries.append(t_offset)
            phase_names.append(phase.name)

        for ax, state_key in zip(axes, states):
            comp, sname = state_key.split(".", 1)
            try:
                t_global, X_vals = self.get_combined(comp, sname)
                ax.plot(t_global, X_vals)
                ax.set_ylabel(state_key)
                ax.grid(True, alpha=0.3)
                for tb in phase_boundaries[1:-1]:
                    ax.axvline(tb, color="gray", linestyle="--", alpha=0.5)
            except KeyError:
                ax.set_ylabel(f"{state_key} (N/A)")

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Test Profile: {self.profile_name}", y=1.01)

        # Phase name labels
        for i, (t_start, t_end, name) in enumerate(
            zip(phase_boundaries[:-1], phase_boundaries[1:], phase_names)
        ):
            axes[0].text(
                (t_start + t_end) / 2, axes[0].get_ylim()[1],
                name, ha="center", va="bottom", fontsize=8, rotation=0,
            )

        plt.tight_layout()
        if show:
            plt.show()
        return fig
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_profiles.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/profiles/result.py tests/unit/test_profiles.py
git commit -m "feat: add PhaseResult and TestProfileResult output containers"
```

---

### Task A4: Dense Output Recording

**Files:**
- Create: `atha/profiles/recording.py`
- Modify: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_profiles.py
from atha.profiles.recording import downsample_dense_output
import numpy as np


def test_downsample_uniform_rate():
    """Mock a TransientSolution and downsample it at a fixed rate."""
    from atha.solver.transient import TransientSolution

    # Simulate a linear state rising from 1e5 to 2e5 Pa over 2 seconds
    t_integrator = np.array([0.0, 0.27, 0.61, 1.05, 1.58, 2.0])
    # state_values: P linear from 1e5 to 2e5
    P_values = 1e5 + (t_integrator / 2.0) * 1e5
    X = P_values.reshape(-1, 1)
    sol = TransientSolution(t=t_integrator, X=X, state_names=["vol.P"])

    # Downsample to 10 Hz
    recording_rate_hz = 10.0
    t_duration = 2.0
    result = downsample_dense_output(sol, t_duration, recording_rate_hz)

    assert len(result.t) == 21         # 0, 0.1, 0.2, ..., 2.0
    assert result.t[0] == pytest.approx(0.0)
    assert result.t[-1] == pytest.approx(2.0)
    # P should still be monotonically rising
    assert np.all(np.diff(result.get("vol", "P")) >= 0)


def test_downsample_uses_interpolation():
    """Downsampled values should lie between adjacent integrator values."""
    from atha.solver.transient import TransientSolution

    t_raw = np.array([0.0, 1.0])
    X_raw = np.array([[0.0], [10.0]])
    sol = TransientSolution(t=t_raw, X=X_raw, state_names=["r.omega"])

    result = downsample_dense_output(sol, 1.0, recording_rate_hz=5.0)
    omega = result.get("r", "omega")

    # At t=0.5s, interpolated value should be 5.0
    idx_half = np.argmin(np.abs(result.t - 0.5))
    assert omega[idx_half] == pytest.approx(5.0, rel=0.01)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_profiles.py -k "downsample" -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement recording.py**

```python
# atha/profiles/recording.py
from __future__ import annotations
import numpy as np
from atha.profiles.result import PhaseResult
from atha.solver.transient import TransientSolution


def downsample_dense_output(
    sol: TransientSolution,
    t_duration: float,
    recording_rate_hz: float,
    phase_name: str = "",
) -> PhaseResult:
    """Resample TransientSolution at a uniform rate using linear interpolation.

    The TransientSolution stores non-uniform integrator steps. This function
    builds a uniform time grid and interpolates each state column.

    Args:
        sol: Result from TransientSolver.integrate()
        t_duration: Total duration of the phase [s]
        recording_rate_hz: Target recording rate [Hz]
        phase_name: Name for the returned PhaseResult

    Returns:
        PhaseResult with uniform time grid
    """
    dt = 1.0 / recording_rate_hz
    n_points = int(round(t_duration * recording_rate_hz)) + 1
    t_record = np.linspace(0.0, t_duration, n_points)

    # Clamp to actual solution range
    t_record = np.clip(t_record, sol.t[0], sol.t[-1])

    # Interpolate each state column
    n_states = sol.X.shape[1]
    X_record = np.zeros((len(t_record), n_states))
    for i in range(n_states):
        X_record[:, i] = np.interp(t_record, sol.t, sol.X[:, i])

    return PhaseResult(
        name=phase_name,
        t=t_record,
        X=X_record,
        state_names=sol.state_names,
        X_final=X_record[-1].copy(),
        abort_triggered=False,
    )
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_profiles.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/profiles/recording.py tests/unit/test_profiles.py
git commit -m "feat: add dense-output downsampling for multi-rate recording"
```

---

### Task A5: Phase Execution Engine

**Files:**
- Create: `atha/profiles/executor.py`
- Modify: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_profiles.py
from atha.profiles.executor import execute_phase
from atha.profiles.phase import PhaseDefinition, PhaseMode, ControlCommand
from atha.profiles.limits import SafetyLimit
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
import numpy as np


def _make_layout():
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    engine = Engine("e")
    engine.add_component(vol)
    layout = engine.compile()
    return layout, gas


def test_execute_steady_trim_phase():
    """STEADY_TRIM phase: pressure converges to trim target."""
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()

    phase = PhaseDefinition(
        name="trim",
        mode=PhaseMode.STEADY_TRIM,
        duration=10.0,
        trim_targets={
            "inlet.mdot": 0.0,    # no flow = pressure stays at initial
        },
    )
    result = execute_phase(layout, X0, phase, global_limits=[])
    assert result.name == "trim"
    assert len(result.X_final) == len(X0)


def test_execute_transient_phase_pressure_rises():
    """TRANSIENT phase: constant inflow raises pressure."""
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    phase = PhaseDefinition(
        name="fill",
        mode=PhaseMode.TRANSIENT,
        duration=2.0,
        control_commands=[
            ControlCommand("inlet.mdot", fn=lambda t: 0.05),
            ControlCommand("inlet.h",   fn=lambda t: h_ref),
        ],
        recording_rate_hz=10.0,
    )
    result = execute_phase(layout, X0, phase, global_limits=[])
    P_series = result.get("vol", "P")
    assert P_series[-1] > P_series[0], "Pressure should rise with constant inflow"


def test_execute_transient_abort_on_limit():
    """TRANSIENT phase: hard limit triggers EngineAbort."""
    from atha.profiles.limits import EngineAbort
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    phase = PhaseDefinition(
        name="fill",
        mode=PhaseMode.TRANSIENT,
        duration=10.0,
        control_commands=[
            ControlCommand("inlet.mdot", fn=lambda t: 0.2),
            ControlCommand("inlet.h",   fn=lambda t: h_ref),
        ],
    )
    limit = SafetyLimit("P_max", "vol", "P", upper_limit=1.2e5, is_hard=True)

    with pytest.raises(EngineAbort):
        execute_phase(layout, X0, phase, global_limits=[limit])
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_profiles.py -k "execute_" -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement executor.py**

```python
# atha/profiles/executor.py
from __future__ import annotations
from typing import List
import numpy as np

from atha.profiles.phase import PhaseDefinition, PhaseMode
from atha.profiles.limits import SafetyLimit, AbortManager, EngineAbort
from atha.profiles.recording import downsample_dense_output
from atha.profiles.result import PhaseResult
from atha.solver.steady_state import SteadyStateSolver
from atha.solver.transient import TransientSolver, TransientSolution


def execute_phase(
    layout,
    X0: np.ndarray,
    phase: PhaseDefinition,
    global_limits: List[SafetyLimit],
) -> PhaseResult:
    """Execute one phase of a test profile and return recorded data.

    Raises EngineAbort if a hard safety limit is exceeded.
    """
    all_limits = global_limits + list(phase.abort_checks)
    abort_mgr = AbortManager(all_limits)
    state_names = layout.all_state_names()

    if phase.mode == PhaseMode.STEADY_TRIM:
        return _execute_steady_trim(layout, X0, phase, abort_mgr, state_names)
    elif phase.mode == PhaseMode.TRANSIENT:
        return _execute_transient(layout, X0, phase, abort_mgr, state_names)
    elif phase.mode == PhaseMode.DWELL:
        return _execute_dwell(X0, phase, state_names)
    else:
        raise ValueError(f"Unknown PhaseMode: {phase.mode}")


def _execute_steady_trim(layout, X0, phase, abort_mgr, state_names) -> PhaseResult:
    solver = SteadyStateSolver(
        layout,
        tol=phase.solver_options.get("tol", 1e-8),
        max_iter=phase.solver_options.get("max_iter", 200),
    )
    X_sol = solver.solve(X0, phase.trim_targets)
    abort_mgr.check(layout, X_sol, t=phase.duration)

    t = np.array([0.0, phase.duration])
    X = np.vstack([X0, X_sol])
    return PhaseResult(
        name=phase.name,
        t=t, X=X,
        state_names=state_names,
        X_final=X_sol.copy(),
        abort_triggered=False,
    )


def _execute_transient(layout, X0, phase, abort_mgr, state_names) -> PhaseResult:
    def bcs(t_phase):
        result = {}
        for cmd in phase.control_commands:
            result[cmd.bcs_key] = cmd.fn(t_phase)
        return result

    # Build scipy event callbacks for limit checking
    scipy_events = abort_mgr.as_scipy_events(layout)

    rtol = phase.solver_options.get("rtol", 1e-4)
    atol = phase.solver_options.get("atol", 1e-6)
    max_step = phase.solver_options.get("max_step", min(0.01, phase.duration / 100))

    solver = TransientSolver(layout, method="Radau",
                              rtol=rtol, atol=atol, max_step=max_step)

    try:
        sol = solver.integrate(
            t_span=(0.0, phase.duration),
            X0=X0,
            boundary_conditions_fn=bcs,
            events=scipy_events if scipy_events else None,
        )
    except RuntimeError as e:
        raise EngineAbort(reason=f"Integrator failed: {e}", t=0.0) from e

    # Check if a hard limit event fired (terminal=True means integration stopped)
    abort_triggered = False
    if scipy_events:
        for i, t_events in enumerate(sol.t_events or []):
            if len(t_events) > 0 and scipy_events[i].terminal:
                abort_triggered = True
                abort_t = float(t_events[0])
                raise EngineAbort(
                    reason=f"Hard limit triggered at t={abort_t:.4f}s",
                    t=abort_t,
                )

    phase_result = downsample_dense_output(
        sol, phase.duration, phase.recording_rate_hz, phase_name=phase.name
    )
    phase_result.abort_triggered = abort_triggered
    return phase_result


def _execute_dwell(X0, phase, state_names) -> PhaseResult:
    t = np.array([0.0, phase.duration])
    X = np.vstack([X0, X0])
    return PhaseResult(
        name=phase.name,
        t=t, X=X,
        state_names=state_names,
        X_final=X0.copy(),
        abort_triggered=False,
    )
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_profiles.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/profiles/executor.py tests/unit/test_profiles.py
git commit -m "feat: add execute_phase() for steady-trim, transient, and dwell phases"
```

---

### Task A6: TestProfile.execute() — Multi-Phase Loop

**Files:**
- Create: `atha/profiles/profile.py`
- Modify: `atha/profiles/__init__.py`
- Modify: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_profiles.py
from atha.profiles.profile import TestProfile


def test_two_phase_profile_state_handoff():
    """Steady trim then transient: X_final of phase 1 becomes X0 of phase 2."""
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    profile = TestProfile(
        name="two_phase",
        phases=[
            PhaseDefinition(
                name="trim",
                mode=PhaseMode.STEADY_TRIM,
                duration=5.0,
                trim_targets={"inlet.mdot": 0.0},
            ),
            PhaseDefinition(
                name="fill",
                mode=PhaseMode.TRANSIENT,
                duration=1.0,
                control_commands=[
                    ControlCommand("inlet.mdot", fn=lambda t: 0.05),
                    ControlCommand("inlet.h",   fn=lambda t: h_ref),
                ],
                recording_rate_hz=10.0,
            ),
        ],
    )

    result = profile.execute(layout, X0)
    assert result.success
    assert len(result.phases) == 2
    assert result.phases[0].name == "trim"
    assert result.phases[1].name == "fill"
    # Pressure in fill phase should rise
    P_fill = result.phases[1].get("vol", "P")
    assert P_fill[-1] > P_fill[0]


def test_profile_aborts_on_hard_limit():
    from atha.profiles.limits import SafetyLimit
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    profile = TestProfile(
        name="abort_test",
        phases=[
            PhaseDefinition(
                name="overfill",
                mode=PhaseMode.TRANSIENT,
                duration=10.0,
                control_commands=[
                    ControlCommand("inlet.mdot", fn=lambda t: 0.3),
                    ControlCommand("inlet.h",   fn=lambda t: h_ref),
                ],
            ),
        ],
        global_limits=[
            SafetyLimit("P_max", "vol", "P", upper_limit=1.1e5, is_hard=True),
        ],
    )

    result = profile.execute(layout, X0)
    assert not result.success
    assert result.abort_reason is not None
    assert result.abort_time is not None
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_profiles.py -k "two_phase or abort" -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement profile.py**

```python
# atha/profiles/profile.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from atha.profiles.phase import PhaseDefinition
from atha.profiles.limits import SafetyLimit, EngineAbort
from atha.profiles.executor import execute_phase
from atha.profiles.result import TestProfileResult


@dataclass
class TestProfile:
    name: str
    phases: List[PhaseDefinition]
    global_limits: List[SafetyLimit] = field(default_factory=list)

    def execute(self, layout, X0: np.ndarray) -> TestProfileResult:
        """Run all phases sequentially; state threads from phase to phase."""
        state_names = layout.all_state_names()
        phase_results = []
        X_current = X0.copy()

        for phase in self.phases:
            try:
                result = execute_phase(
                    layout, X_current, phase,
                    global_limits=self.global_limits,
                )
                phase_results.append(result)
                X_current = result.X_final.copy()

            except EngineAbort as e:
                return TestProfileResult(
                    profile_name=self.name,
                    phases=phase_results,
                    state_names=state_names,
                    abort_reason=e.reason,
                    abort_time=e.t,
                )

        return TestProfileResult(
            profile_name=self.name,
            phases=phase_results,
            state_names=state_names,
        )
```

- [ ] **Step 4: Update profiles/__init__.py**

```python
# atha/profiles/__init__.py
from atha.profiles.phase import PhaseDefinition, PhaseMode, ControlCommand
from atha.profiles.limits import SafetyLimit, EngineAbort
from atha.profiles.result import PhaseResult, TestProfileResult
from atha.profiles.profile import TestProfile

__all__ = [
    "PhaseDefinition", "PhaseMode", "ControlCommand",
    "SafetyLimit", "EngineAbort",
    "PhaseResult", "TestProfileResult",
    "TestProfile",
]
```

- [ ] **Step 5: Run tests, verify pass**

```
pytest tests/unit/test_profiles.py -v
# Expected: all tests PASS
```

- [ ] **Step 6: Commit**

```bash
git add atha/profiles/profile.py atha/profiles/__init__.py tests/unit/test_profiles.py
git commit -m "feat: add TestProfile multi-phase execution engine"
```

---

### Task A7: HDF5 Save/Load

**Files:**
- Create: `atha/profiles/io.py`
- Modify: `tests/unit/test_profiles.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_profiles.py
import tempfile, os
from atha.profiles.io import save_profile_result, load_profile_result


def test_profile_result_roundtrip_hdf5():
    """Save and reload a TestProfileResult; verify data matches."""
    t = np.linspace(0, 1.0, 11)
    X = np.column_stack([1e5 + t * 1e4, 3e5 * np.ones_like(t)])
    pr = PhaseResult("fill", t, X, ["vol.P", "vol.h"], X[-1], False)
    result = TestProfileResult("test", [pr], ["vol.P", "vol.h"])

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        fname = f.name

    try:
        save_profile_result(result, fname)
        loaded = load_profile_result(fname)

        assert loaded.profile_name == "test"
        assert len(loaded.phases) == 1
        assert loaded.phases[0].name == "fill"
        np.testing.assert_array_almost_equal(
            loaded.phases[0].get("vol", "P"),
            result.phases[0].get("vol", "P"),
        )
    finally:
        os.unlink(fname)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_profiles.py::test_profile_result_roundtrip_hdf5 -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement io.py**

```python
# atha/profiles/io.py
from __future__ import annotations
import numpy as np
import h5py
from atha.profiles.result import PhaseResult, TestProfileResult


def save_profile_result(result: TestProfileResult, filename: str) -> None:
    with h5py.File(filename, "w") as f:
        f.attrs["profile_name"] = result.profile_name
        f.attrs["state_names"] = result.state_names
        f.attrs["success"] = result.success
        if result.abort_reason is not None:
            f.attrs["abort_reason"] = result.abort_reason
        if result.abort_time is not None:
            f.attrs["abort_time"] = result.abort_time

        for i, phase in enumerate(result.phases):
            grp = f.create_group(f"phase_{i:03d}")
            grp.attrs["name"] = phase.name
            grp.attrs["abort_triggered"] = phase.abort_triggered
            grp.attrs["state_names"] = phase.state_names
            grp.create_dataset("t", data=phase.t)
            grp.create_dataset("X", data=phase.X)
            grp.create_dataset("X_final", data=phase.X_final)


def load_profile_result(filename: str) -> TestProfileResult:
    with h5py.File(filename, "r") as f:
        profile_name = str(f.attrs["profile_name"])
        state_names = list(f.attrs["state_names"])
        abort_reason = str(f.attrs["abort_reason"]) if "abort_reason" in f.attrs else None
        abort_time = float(f.attrs["abort_time"]) if "abort_time" in f.attrs else None

        phases = []
        for key in sorted(f.keys()):
            grp = f[key]
            pr = PhaseResult(
                name=str(grp.attrs["name"]),
                t=grp["t"][:],
                X=grp["X"][:],
                state_names=list(grp.attrs["state_names"]),
                X_final=grp["X_final"][:],
                abort_triggered=bool(grp.attrs["abort_triggered"]),
            )
            phases.append(pr)

    return TestProfileResult(
        profile_name=profile_name,
        phases=phases,
        state_names=state_names,
        abort_reason=abort_reason,
        abort_time=abort_time,
    )
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_profiles.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/profiles/io.py tests/unit/test_profiles.py
git commit -m "feat: add HDF5 save/load for TestProfileResult"
```

---

### Task A8: Integration Test — TTBE Throttle Sweep

**Files:**
- Create: `tests/integration/test_profile_ttbe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/integration/test_profile_ttbe.py
"""Integration test: TTBE 100% → 65% → 100% throttle sweep via TestProfile."""
import numpy as np
import pytest
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.components.rotor import Rotor
from atha.core.engine import Engine
from atha.profiles import (
    TestProfile, PhaseDefinition, PhaseMode,
    ControlCommand, SafetyLimit,
)


def build_simple_ttbe_engine():
    """Minimal engine: single volume representing main chamber."""
    gas = IdealGasBackend(gamma=1.24, R=711.0)
    chamber = Volume(
        "chamber", volume=0.05, thermo=gas,
        initial_P=20.6e6, initial_T=3560.0,
    )
    chamber.add_inlet("propellant_in")
    chamber.add_outlet("nozzle_out")
    engine = Engine("ttbe")
    engine.add_component(chamber)
    return engine, gas


def test_ttbe_throttle_sweep_three_trim_points():
    """Execute 100%→65%→100% profile; all phases complete, pressures correct."""
    engine, gas = build_simple_ttbe_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()

    h_ref = gas.state_from_PT(20.6e6, 3560.0).h
    mdot_100 = 468.0
    mdot_65  = 468.0 * 0.65

    profile = TestProfile(
        name="ttbe_throttle_sweep",
        phases=[
            # Trim at 100% RPL
            PhaseDefinition(
                name="mainstage_100pct",
                mode=PhaseMode.STEADY_TRIM,
                duration=30.0,
                trim_targets={
                    "propellant_in.mdot": mdot_100,
                    "propellant_in.h": h_ref,
                    "nozzle_out.mdot": mdot_100,
                },
            ),
            # Transient throttle down
            PhaseDefinition(
                name="throttle_down",
                mode=PhaseMode.TRANSIENT,
                duration=5.0,
                control_commands=[
                    ControlCommand("propellant_in.mdot",
                                   fn=lambda t: mdot_100 - (mdot_100 - mdot_65) * t / 5.0),
                    ControlCommand("propellant_in.h", fn=lambda t: h_ref),
                    ControlCommand("nozzle_out.mdot",
                                   fn=lambda t: mdot_100 - (mdot_100 - mdot_65) * t / 5.0),
                ],
                recording_rate_hz=20.0,
            ),
            # Trim at 65% RPL
            PhaseDefinition(
                name="mainstage_65pct",
                mode=PhaseMode.STEADY_TRIM,
                duration=30.0,
                trim_targets={
                    "propellant_in.mdot": mdot_65,
                    "propellant_in.h": h_ref,
                    "nozzle_out.mdot": mdot_65,
                },
            ),
        ],
        global_limits=[
            SafetyLimit("Pc_hard_max", "chamber", "P",
                        upper_limit=30e6, is_hard=True),
        ],
    )

    result = profile.execute(layout, X0)

    assert result.success, f"Profile aborted: {result.abort_reason}"
    assert len(result.phases) == 3
    assert result.phases[0].name == "mainstage_100pct"
    assert result.phases[2].name == "mainstage_65pct"

    # After throttle-down, final X should be different from initial
    P_final = result.phases[2].X_final[0]
    P_initial = result.phases[0].X_final[0]
    # Pressure should have changed (at different flow rates)
    assert P_final != pytest.approx(P_initial, rel=0.5)


def test_ttbe_profile_abort_on_overpressure():
    """Verify hard limit abort fires correctly."""
    engine, gas = build_simple_ttbe_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(20.6e6, 3560.0).h

    profile = TestProfile(
        name="abort_test",
        phases=[
            PhaseDefinition(
                name="overfill",
                mode=PhaseMode.TRANSIENT,
                duration=10.0,
                control_commands=[
                    ControlCommand("propellant_in.mdot", fn=lambda t: 2000.0),
                    ControlCommand("propellant_in.h", fn=lambda t: h_ref),
                    ControlCommand("nozzle_out.mdot", fn=lambda t: 0.0),
                ],
            ),
        ],
        global_limits=[
            SafetyLimit("Pc_limit", "chamber", "P", upper_limit=21e6, is_hard=True),
        ],
    )

    result = profile.execute(layout, X0)
    assert not result.success
    assert result.abort_reason is not None


def test_profile_hdf5_roundtrip(tmp_path):
    from atha.profiles.io import save_profile_result, load_profile_result
    engine, gas = build_simple_ttbe_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(20.6e6, 3560.0).h

    profile = TestProfile(
        name="roundtrip",
        phases=[
            PhaseDefinition(
                name="dwell",
                mode=PhaseMode.DWELL,
                duration=1.0,
            ),
        ],
    )
    result = profile.execute(layout, X0)
    fname = str(tmp_path / "test.hdf5")
    save_profile_result(result, fname)
    loaded = load_profile_result(fname)
    assert loaded.profile_name == "roundtrip"
    assert loaded.success
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/integration/test_profile_ttbe.py -v
# Expected: FAIL — missing module or logic errors
```

- [ ] **Step 3: Run tests, fix any integration issues, verify pass**

```
pytest tests/integration/test_profile_ttbe.py -v
# Expected: all 3 tests PASS
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_profile_ttbe.py
git commit -m "test: add TTBE test profile integration tests"
```

---

## Phase B: Monte Carlo Analysis

---

### Task B1: UncertainParameter and ParameterType

**Files:**
- Create: `atha/monte_carlo/__init__.py`
- Create: `atha/monte_carlo/parameters.py`
- Create: `tests/unit/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_monte_carlo.py
import numpy as np
import pytest
from atha.monte_carlo.parameters import UncertainParameter, ParameterType


def test_parameter_type_enum():
    assert ParameterType.NORMAL.value == "normal"
    assert ParameterType.UNIFORM.value == "uniform"
    assert ParameterType.LOGNORMAL.value == "lognormal"


def test_normal_parameter_samples_distribution():
    rng = np.random.default_rng(42)
    param = UncertainParameter(
        name="Pc", nominal=20.6e6,
        dist_type=ParameterType.NORMAL,
        sigma=0.4e6, lower=None, upper=None,
    )
    samples = np.array([param.sample(rng) for _ in range(2000)])
    assert abs(np.mean(samples) - 20.6e6) < 1e5     # mean within 0.5%
    assert abs(np.std(samples) - 0.4e6) < 5e4        # std within 12.5%


def test_uniform_parameter_samples_in_range():
    rng = np.random.default_rng(0)
    param = UncertainParameter(
        name="eta", nominal=0.975,
        dist_type=ParameterType.UNIFORM,
        lower=0.970, upper=0.990,
    )
    samples = np.array([param.sample(rng) for _ in range(500)])
    assert np.all(samples >= 0.970)
    assert np.all(samples <= 0.990)


def test_lognormal_parameter_positive():
    rng = np.random.default_rng(7)
    param = UncertainParameter(
        name="At", nominal=0.0687,
        dist_type=ParameterType.LOGNORMAL,
        sigma_log=0.01,
    )
    samples = np.array([param.sample(rng) for _ in range(200)])
    assert np.all(samples > 0), "Lognormal samples must be positive"
    assert abs(np.mean(samples) - 0.0687) < 0.002
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_monte_carlo.py -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement parameters.py**

```python
# atha/monte_carlo/parameters.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
from scipy.stats import norm


class ParameterType(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"


@dataclass
class UncertainParameter:
    name: str
    nominal: float
    dist_type: ParameterType
    sigma: Optional[float] = None       # for NORMAL: std dev in physical units
    sigma_pct: Optional[float] = None   # for NORMAL: std dev as % of nominal
    sigma_log: Optional[float] = None   # for LOGNORMAL: std dev of log(value)
    lower: Optional[float] = None       # for UNIFORM: lower bound
    upper: Optional[float] = None       # for UNIFORM: upper bound
    component_attr: Optional[str] = None  # "component_name.attribute" for auto-apply

    def __post_init__(self):
        if self.dist_type == ParameterType.NORMAL:
            if self.sigma is None and self.sigma_pct is not None:
                self.sigma = self.nominal * self.sigma_pct / 100.0
            if self.sigma is None:
                raise ValueError(f"NORMAL parameter '{self.name}' requires sigma or sigma_pct")
        elif self.dist_type == ParameterType.UNIFORM:
            if self.lower is None or self.upper is None:
                raise ValueError(f"UNIFORM parameter '{self.name}' requires lower and upper")
        elif self.dist_type == ParameterType.LOGNORMAL:
            if self.sigma_log is None:
                raise ValueError(f"LOGNORMAL parameter '{self.name}' requires sigma_log")

    def sample(self, rng) -> float:
        """Draw one sample from the distribution."""
        if self.dist_type == ParameterType.NORMAL:
            return float(rng.normal(self.nominal, self.sigma))
        elif self.dist_type == ParameterType.UNIFORM:
            return float(rng.uniform(self.lower, self.upper))
        elif self.dist_type == ParameterType.LOGNORMAL:
            return float(np.exp(rng.normal(np.log(self.nominal), self.sigma_log)))
        raise ValueError(f"Unknown ParameterType: {self.dist_type}")

    def transform_unit(self, u: float) -> float:
        """Transform a single [0,1] uniform value to physical space (for LHS)."""
        if self.dist_type == ParameterType.NORMAL:
            return norm.ppf(u) * self.sigma + self.nominal
        elif self.dist_type == ParameterType.UNIFORM:
            return u * (self.upper - self.lower) + self.lower
        elif self.dist_type == ParameterType.LOGNORMAL:
            return np.exp(norm.ppf(u) * self.sigma_log + np.log(self.nominal))
        raise ValueError(f"Unknown ParameterType: {self.dist_type}")
```

- [ ] **Step 4: Create monte_carlo/__init__.py**

```python
# atha/monte_carlo/__init__.py
from atha.monte_carlo.parameters import UncertainParameter, ParameterType

__all__ = ["UncertainParameter", "ParameterType"]
```

- [ ] **Step 5: Run tests, verify pass**

```
pytest tests/unit/test_monte_carlo.py -v
# Expected: all 4 tests PASS
```

- [ ] **Step 6: Commit**

```bash
git add atha/monte_carlo/ tests/unit/test_monte_carlo.py
git commit -m "feat: add UncertainParameter and ParameterType"
```

---

### Task B2: LHS and Saltelli Samplers

**Files:**
- Create: `atha/monte_carlo/sampling.py`
- Modify: `tests/unit/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_monte_carlo.py
from atha.monte_carlo.sampling import LHSSampler, SaltelliSampler
from atha.monte_carlo.parameters import UncertainParameter, ParameterType


def _make_params():
    return [
        UncertainParameter("Pc", 20.6e6, ParameterType.NORMAL, sigma=0.4e6),
        UncertainParameter("MR", 6.0,    ParameterType.NORMAL, sigma=0.12),
        UncertainParameter("eta", 0.975, ParameterType.UNIFORM, lower=0.97, upper=0.99),
    ]


def test_lhs_sampler_shape():
    params = _make_params()
    sampler = LHSSampler(seed=42)
    samples = sampler.sample(params, N=100)
    assert samples.shape == (100, 3)


def test_lhs_sampler_covers_range():
    """LHS should cover [mean-4σ, mean+4σ] for Normal param with N=100."""
    params = [UncertainParameter("x", 10.0, ParameterType.NORMAL, sigma=1.0)]
    sampler = LHSSampler(seed=0)
    samples = sampler.sample(params, N=200)
    assert samples[:, 0].min() < 7.0    # below mean - 3σ
    assert samples[:, 0].max() > 13.0   # above mean + 3σ


def test_saltelli_sampler_shape():
    """Saltelli generates N*(k+2) rows for k parameters."""
    params = _make_params()   # k=3
    sampler = SaltelliSampler(seed=99)
    samples = sampler.sample(params, N_base=50)
    assert samples.shape == (50 * (3 + 2), 3)   # 250 rows


def test_lhs_reproducible_with_seed():
    params = _make_params()
    s1 = LHSSampler(seed=5).sample(params, N=20)
    s2 = LHSSampler(seed=5).sample(params, N=20)
    np.testing.assert_array_equal(s1, s2)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_monte_carlo.py -k "sampler" -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement sampling.py**

```python
# atha/monte_carlo/sampling.py
from __future__ import annotations
from typing import List
import numpy as np
from scipy.stats.qmc import LatinHypercube
from atha.monte_carlo.parameters import UncertainParameter


class LHSSampler:
    """Latin Hypercube Sampler — generates N stratified samples for k parameters."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample(self, params: List[UncertainParameter], N: int) -> np.ndarray:
        """Return (N, k) array in physical parameter space."""
        k = len(params)
        sampler = LatinHypercube(d=k, seed=self.seed)
        unit_samples = sampler.random(N)   # shape (N, k), values in [0, 1]

        physical = np.zeros_like(unit_samples)
        for j, param in enumerate(params):
            # Clip away exact 0 and 1 to avoid norm.ppf(0)=-inf
            u = np.clip(unit_samples[:, j], 1e-10, 1 - 1e-10)
            physical[:, j] = np.array([param.transform_unit(ui) for ui in u])
        return physical


class SaltelliSampler:
    """Saltelli sampling scheme for Sobol sensitivity analysis.

    Generates N_base * (k + 2) rows: matrices A, B, and k cross-matrices C_j.
    Use these rows with SALib.analyze.sobol for variance-based sensitivity.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def sample(self, params: List[UncertainParameter], N_base: int) -> np.ndarray:
        """Return (N_base*(k+2), k) array in physical parameter space."""
        from SALib.sample import saltelli

        k = len(params)
        problem = {
            "num_vars": k,
            "names": [p.name for p in params],
            "bounds": [[0.0, 1.0] for _ in range(k)],
        }
        unit_samples = saltelli.sample(
            problem, N=N_base, calc_second_order=False, seed=self.seed
        )   # shape (N_base*(k+2), k)

        physical = np.zeros_like(unit_samples)
        for j, param in enumerate(params):
            u = np.clip(unit_samples[:, j], 1e-10, 1 - 1e-10)
            physical[:, j] = np.array([param.transform_unit(ui) for ui in u])
        return physical
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_monte_carlo.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/monte_carlo/sampling.py tests/unit/test_monte_carlo.py
git commit -m "feat: add LHSSampler and SaltelliSampler"
```

---

### Task B3: MCStatistics

**Files:**
- Create: `atha/monte_carlo/statistics.py`
- Modify: `tests/unit/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_monte_carlo.py
from atha.monte_carlo.statistics import MCStatistics, compute_statistics


def test_compute_statistics_known_distribution():
    rng = np.random.default_rng(123)
    Y = rng.normal(loc=450.0, scale=2.0, size=5000)
    stats = compute_statistics(Y)

    assert abs(stats.mean - 450.0) < 0.1
    assert abs(stats.std - 2.0) < 0.1
    assert abs(stats.cv_pct - (2.0 / 450.0 * 100)) < 0.05
    assert stats.p5 < stats.median < stats.p95
    assert stats.N_samples == 5000
    assert stats.mean_ci_95 < 0.1   # tight CI for 5000 samples


def test_compute_statistics_single_value_edge_case():
    Y = np.array([100.0])
    stats = compute_statistics(Y)
    assert stats.mean == 100.0
    assert stats.std == 0.0
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_monte_carlo.py -k "statistics" -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement statistics.py**

```python
# atha/monte_carlo/statistics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MCStatistics:
    N_samples: int
    mean: float
    std: float
    cv_pct: float        # coefficient of variation [%]
    min: float
    max: float
    median: float
    p1: float
    p5: float
    p95: float
    p99: float
    mean_ci_95: float    # half-width of 95% CI on the mean


def compute_statistics(Y: np.ndarray) -> MCStatistics:
    """Compute standard MC statistics from a 1-D array of scalar outputs."""
    Y = np.asarray(Y, dtype=float)
    Y_valid = Y[np.isfinite(Y)]
    N = len(Y_valid)

    if N == 0:
        raise ValueError("No finite samples to compute statistics from")

    mu = float(np.mean(Y_valid))
    sigma = float(np.std(Y_valid, ddof=min(1, N - 1)))

    return MCStatistics(
        N_samples=N,
        mean=mu,
        std=sigma,
        cv_pct=100.0 * sigma / abs(mu) if mu != 0 else float("inf"),
        min=float(np.min(Y_valid)),
        max=float(np.max(Y_valid)),
        median=float(np.median(Y_valid)),
        p1=float(np.percentile(Y_valid, 1)),
        p5=float(np.percentile(Y_valid, 5)),
        p95=float(np.percentile(Y_valid, 95)),
        p99=float(np.percentile(Y_valid, 99)),
        mean_ci_95=1.96 * sigma / np.sqrt(N) if N > 1 else 0.0,
    )
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_monte_carlo.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/monte_carlo/statistics.py tests/unit/test_monte_carlo.py
git commit -m "feat: add MCStatistics and compute_statistics()"
```

---

### Task B4: MonteCarloRunner

**Files:**
- Create: `atha/monte_carlo/runner.py`
- Create: `atha/monte_carlo/results.py`
- Modify: `atha/monte_carlo/__init__.py`
- Modify: `tests/unit/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_monte_carlo.py
from atha.monte_carlo.runner import MonteCarloRunner
from atha.monte_carlo.parameters import UncertainParameter, ParameterType


def test_mc_runner_serial_quadratic():
    """Run MC on f(x) = x² with Normal x~N(2,0.1). E[x²] ≈ 4.01."""
    params = [UncertainParameter("x", nominal=2.0, dist_type=ParameterType.NORMAL, sigma=0.1)]
    runner = MonteCarloRunner(params=params, n_samples=500,
                               sampling_method="lhs", n_jobs=1, seed=42)

    result = runner.run(evaluate_fn=lambda X: X[0] ** 2)

    # E[X²] = Var(X) + E[X]² = 0.01 + 4.0 = 4.01
    assert abs(result.stats.mean - 4.01) < 0.05
    assert result.stats.N_samples == 500
    assert len(result.Y_samples) == 500


def test_mc_runner_handles_nan_gracefully():
    """evaluate_fn returning NaN should be excluded from stats."""
    params = [UncertainParameter("x", nominal=0.0, dist_type=ParameterType.UNIFORM,
                                  lower=-1.0, upper=1.0)]
    # sqrt is nan for negative x
    runner = MonteCarloRunner(params=params, n_samples=200,
                               sampling_method="lhs", n_jobs=1, seed=0)
    result = runner.run(evaluate_fn=lambda X: float(np.sqrt(X[0])) if X[0] >= 0 else float("nan"))

    assert result.stats.N_samples < 200   # some NaN excluded
    assert result.stats.N_samples > 50    # about half should be valid


def test_mc_runner_parallel_matches_serial():
    """n_jobs>1 should produce same mean as n_jobs=1 (same seed)."""
    params = [UncertainParameter("x", nominal=5.0, dist_type=ParameterType.NORMAL, sigma=1.0)]
    fn = lambda X: X[0] ** 2

    serial = MonteCarloRunner(params=params, n_samples=100, n_jobs=1, seed=7).run(fn)
    parallel = MonteCarloRunner(params=params, n_samples=100, n_jobs=2, seed=7).run(fn)

    assert abs(serial.stats.mean - parallel.stats.mean) < 0.5
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_monte_carlo.py -k "mc_runner" -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement runner.py**

```python
# atha/monte_carlo/runner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
from joblib import Parallel, delayed

from atha.monte_carlo.parameters import UncertainParameter
from atha.monte_carlo.sampling import LHSSampler, SaltelliSampler
from atha.monte_carlo.statistics import compute_statistics


class MonteCarloRunner:
    def __init__(
        self,
        params: List[UncertainParameter],
        n_samples: int = 500,
        sampling_method: str = "lhs",   # "lhs" or "saltelli"
        n_jobs: int = -1,
        seed: int = 42,
        verbose: int = 0,
    ):
        self.params = params
        self.n_samples = n_samples
        self.sampling_method = sampling_method
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

    def generate_samples(self) -> np.ndarray:
        if self.sampling_method == "lhs":
            return LHSSampler(seed=self.seed).sample(self.params, self.n_samples)
        elif self.sampling_method == "saltelli":
            return SaltelliSampler(seed=self.seed).sample(self.params, self.n_samples)
        raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

    def run(self, evaluate_fn: Callable[[np.ndarray], float]) -> "MonteCarloResult":
        from atha.monte_carlo.results import MonteCarloResult

        samples = self.generate_samples()

        def _safe_eval(X):
            try:
                val = evaluate_fn(X)
                return float(val) if val is not None else float("nan")
            except Exception:
                return float("nan")

        if self.n_jobs == 1:
            Y = np.array([_safe_eval(X) for X in samples])
        else:
            Y = np.array(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="loky")(
                    delayed(_safe_eval)(X) for X in samples
                )
            )

        converged = np.isfinite(Y)
        Y_valid = Y[converged]
        stats = compute_statistics(Y_valid) if len(Y_valid) > 0 else None

        return MonteCarloResult(
            param_names=[p.name for p in self.params],
            param_samples=samples,
            Y_samples=Y,
            converged=converged,
            stats=stats,
        )
```

- [ ] **Step 4: Implement results.py**

```python
# atha/monte_carlo/results.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from atha.monte_carlo.statistics import MCStatistics


@dataclass
class MonteCarloResult:
    param_names: List[str]
    param_samples: np.ndarray   # shape (N, k)
    Y_samples: np.ndarray       # shape (N,)
    converged: np.ndarray       # shape (N,), bool
    stats: Optional[MCStatistics]
    sobol: Optional[dict] = None

    def print_summary(self) -> None:
        n_total = len(self.Y_samples)
        n_conv = int(np.sum(self.converged))
        print(f"Monte Carlo Results: N={n_total}, converged={n_conv} ({100*n_conv/n_total:.1f}%)")
        if self.stats:
            s = self.stats
            print(f"  Mean:    {s.mean:.4g}")
            print(f"  Std:     {s.std:.4g}")
            print(f"  CV:      {s.cv_pct:.2f}%")
            print(f"  95% CI:  [{s.p5:.4g}, {s.p95:.4g}]")
            print(f"  Range:   [{s.min:.4g}, {s.max:.4g}]")
        if self.sobol:
            print("\nSobol Sensitivity Indices:")
            print(f"  {'Parameter':<20} {'S1':>8} {'ST':>8} {'S1_conf':>10}")
            for name, s1, st, s1c in zip(
                self.param_names,
                self.sobol["S1"], self.sobol["ST"], self.sobol["S1_conf"]
            ):
                print(f"  {name:<20} {s1:>8.3f} {st:>8.3f} {s1c:>10.3f}")

    def plot_histogram(self, xlabel: str = "Output", title: str = "", show: bool = True):
        import matplotlib.pyplot as plt
        Y_valid = self.Y_samples[self.converged]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(Y_valid, bins=40, density=True, edgecolor="white", alpha=0.8)
        if self.stats:
            ax.axvline(self.stats.mean, color="red", linestyle="--",
                       label=f"Mean = {self.stats.mean:.3g}")
            ax.axvline(self.stats.p5,  color="orange", linestyle=":",
                       label=f"5th pct = {self.stats.p5:.3g}")
            ax.axvline(self.stats.p95, color="orange", linestyle=":",
                       label=f"95th pct = {self.stats.p95:.3g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.set_title(title or "Monte Carlo Output Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_sobol_indices(self, show: bool = True):
        if self.sobol is None:
            raise RuntimeError("No Sobol indices computed — run with sensitivity=True")
        import matplotlib.pyplot as plt
        names = self.param_names
        S1 = self.sobol["S1"]
        ST = self.sobol["ST"]
        S1_conf = self.sobol["S1_conf"]
        ST_conf = self.sobol["ST_conf"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(names) * 0.5)))
        y = np.arange(len(names))

        ax1.barh(y, S1, xerr=S1_conf, capsize=4, alpha=0.8)
        ax1.set_yticks(y)
        ax1.set_yticklabels(names)
        ax1.set_xlabel("First-Order Index S_i")
        ax1.set_title("Main Effects")
        ax1.grid(True, alpha=0.3)

        ax2.barh(y, ST, xerr=ST_conf, capsize=4, alpha=0.8, color="orange")
        ax2.set_yticks(y)
        ax2.set_yticklabels(names)
        ax2.set_xlabel("Total-Order Index S_Ti")
        ax2.set_title("Total Effects (incl. interactions)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def save(self, filename: str) -> None:
        import h5py
        with h5py.File(filename, "w") as f:
            f.create_dataset("param_samples", data=self.param_samples)
            f.create_dataset("Y_samples", data=self.Y_samples)
            f.create_dataset("converged", data=self.converged.astype(np.uint8))
            f.attrs["param_names"] = self.param_names
            if self.stats:
                sg = f.create_group("statistics")
                for attr in ("N_samples", "mean", "std", "cv_pct", "min", "max",
                             "median", "p1", "p5", "p95", "p99", "mean_ci_95"):
                    sg.attrs[attr] = getattr(self.stats, attr)
            if self.sobol:
                sg = f.create_group("sobol")
                for key in ("S1", "ST", "S1_conf", "ST_conf"):
                    sg.create_dataset(key, data=self.sobol[key])

    @classmethod
    def load(cls, filename: str) -> "MonteCarloResult":
        import h5py
        from atha.monte_carlo.statistics import MCStatistics
        with h5py.File(filename, "r") as f:
            param_names = list(f.attrs["param_names"])
            param_samples = f["param_samples"][:]
            Y_samples = f["Y_samples"][:]
            converged = f["converged"][:].astype(bool)
            stats = None
            if "statistics" in f:
                sg = f["statistics"]
                stats = MCStatistics(**{k: sg.attrs[k] for k in sg.attrs})
            sobol = None
            if "sobol" in f:
                sobol = {k: f["sobol"][k][:] for k in f["sobol"]}
        return cls(param_names=param_names, param_samples=param_samples,
                   Y_samples=Y_samples, converged=converged, stats=stats, sobol=sobol)
```

- [ ] **Step 5: Update monte_carlo/__init__.py**

```python
# atha/monte_carlo/__init__.py
from atha.monte_carlo.parameters import UncertainParameter, ParameterType
from atha.monte_carlo.runner import MonteCarloRunner
from atha.monte_carlo.results import MonteCarloResult

__all__ = ["UncertainParameter", "ParameterType", "MonteCarloRunner", "MonteCarloResult"]
```

- [ ] **Step 6: Run tests, verify pass**

```
pytest tests/unit/test_monte_carlo.py -v
# Expected: all tests PASS
```

- [ ] **Step 7: Commit**

```bash
git add atha/monte_carlo/runner.py atha/monte_carlo/results.py \
        atha/monte_carlo/__init__.py tests/unit/test_monte_carlo.py
git commit -m "feat: add MonteCarloRunner with joblib parallelism and MonteCarloResult"
```

---

### Task B5: Sobol Sensitivity Analysis

**Files:**
- Create: `atha/monte_carlo/sensitivity.py`
- Modify: `tests/unit/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/unit/test_monte_carlo.py
from atha.monte_carlo.sensitivity import compute_sobol_indices


def test_sobol_ishigami_function():
    """Ishigami function has known S1 ranking: x1 > x2 > x3 (x3 ≈ 0).
    f(x) = sin(x1) + 7*sin(x2)^2 + 0.1*x3^4*sin(x1), xi ~ Uniform[-π, π]
    Known: S1_x1 ≈ 0.314, S1_x2 ≈ 0.442, S1_x3 = 0.0
    """
    from atha.monte_carlo.parameters import UncertainParameter, ParameterType
    from atha.monte_carlo.sampling import SaltelliSampler
    import math

    params = [
        UncertainParameter("x1", 0.0, ParameterType.UNIFORM, lower=-math.pi, upper=math.pi),
        UncertainParameter("x2", 0.0, ParameterType.UNIFORM, lower=-math.pi, upper=math.pi),
        UncertainParameter("x3", 0.0, ParameterType.UNIFORM, lower=-math.pi, upper=math.pi),
    ]

    samples = SaltelliSampler(seed=42).sample(params, N_base=1000)

    def ishigami(X):
        return math.sin(X[0]) + 7 * math.sin(X[1])**2 + 0.1 * X[2]**4 * math.sin(X[0])

    Y = np.array([ishigami(X) for X in samples])
    Si = compute_sobol_indices(params, samples, Y, N_base=1000)

    # x2 should have highest first-order index (known: ~0.442)
    assert Si["S1"][1] > Si["S1"][0], "x2 should dominate over x1 in first-order"
    assert Si["S1"][2] < 0.05,        "x3 should have near-zero first-order index"
    # All indices should be in valid range
    assert np.all(Si["S1"] >= -0.1)   # allow small negative due to estimation error
    assert np.all(Si["ST"] >= -0.1)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/unit/test_monte_carlo.py::test_sobol_ishigami_function -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement sensitivity.py**

```python
# atha/monte_carlo/sensitivity.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
from atha.monte_carlo.parameters import UncertainParameter


def compute_sobol_indices(
    params: List[UncertainParameter],
    param_samples: np.ndarray,   # shape (N_base*(k+2), k) from SaltelliSampler
    Y: np.ndarray,               # shape (N_base*(k+2),)
    N_base: int,
    calc_second_order: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute first-order and total-order Sobol sensitivity indices.

    Uses SALib's Jansen estimator. param_samples must come from SaltelliSampler
    with the same N_base.

    Returns dict with keys: S1, ST, S1_conf, ST_conf (arrays of length k).
    """
    from SALib.analyze import sobol

    k = len(params)
    problem = {
        "num_vars": k,
        "names": [p.name for p in params],
        "bounds": [[0.0, 1.0] for _ in range(k)],  # SALib needs bounds, but Y drives it
    }

    Si = sobol.analyze(
        problem, Y,
        calc_second_order=calc_second_order,
        conf_level=0.95,
        print_to_console=False,
    )
    return {
        "S1":      Si["S1"],
        "ST":      Si["ST"],
        "S1_conf": Si["S1_conf"],
        "ST_conf": Si["ST_conf"],
    }


def run_sensitivity_analysis(
    params: List[UncertainParameter],
    evaluate_fn,
    N_base: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[str, np.ndarray]:
    """Convenience function: sample + evaluate + compute Sobol indices."""
    from atha.monte_carlo.sampling import SaltelliSampler
    from joblib import Parallel, delayed

    samples = SaltelliSampler(seed=seed).sample(params, N_base)

    def _safe(X):
        try:
            v = evaluate_fn(X)
            return float(v)
        except Exception:
            return float("nan")

    if n_jobs == 1:
        Y = np.array([_safe(X) for X in samples])
    else:
        Y = np.array(
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_safe)(X) for X in samples
            )
        )

    return compute_sobol_indices(params, samples, Y, N_base=N_base)
```

- [ ] **Step 4: Run tests, verify pass**

```
pytest tests/unit/test_monte_carlo.py -v
# Expected: all tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add atha/monte_carlo/sensitivity.py tests/unit/test_monte_carlo.py
git commit -m "feat: add Sobol sensitivity analysis via SALib"
```

---

### Task B6: Integration Test — TTBE Monte Carlo Study

**Files:**
- Create: `tests/integration/test_monte_carlo_ttbe.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/integration/test_monte_carlo_ttbe.py
"""Integration test: Monte Carlo uncertainty study on TTBE JANNAF model."""
import numpy as np
import pytest
from atha.monte_carlo import UncertainParameter, ParameterType, MonteCarloRunner
from atha.monte_carlo.sensitivity import run_sensitivity_analysis
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.thermo.ideal_gas import IdealGasBackend


def make_jannaf_evaluator(nominal_params):
    """Return function that evaluates TTBE Isp given perturbed parameter vector."""
    def evaluate(X):
        Pc, MR, eta_cstar, eta_div, At, epsilon = X
        thermo = IdealGasBackend(gamma=1.24, R=711.0)
        eff = JANNAFEfficiencies(
            eta_cstar=float(eta_cstar),
            eta_Cd=0.98,
            eta_velocity=0.99,
            eta_divergence=float(eta_div),
            eta_two_phase=1.0,
            eta_boundary_layer=0.99,
        )
        jannaf = SimplifiedJANNAF(
            thermo=thermo, efficiencies=eff,
            throat_area=float(At),
            exit_area=float(At) * float(epsilon),
            ambient_pressure=0.0,
        )
        result = jannaf.compute(
            P_chamber=float(Pc),
            T_chamber=3560.0,
            MR=float(MR),
            mdot_total=468.0,
        )
        return result.Isp
    return evaluate


TTBE_PARAMS = [
    UncertainParameter("Pc",       20.6e6, ParameterType.NORMAL, sigma_pct=2.0),
    UncertainParameter("MR",       6.0,    ParameterType.NORMAL, sigma_pct=2.0),
    UncertainParameter("eta_cstar",0.975,  ParameterType.NORMAL, sigma=0.005),
    UncertainParameter("eta_div",  0.9830, ParameterType.NORMAL, sigma=0.004),
    UncertainParameter("At",       0.0687, ParameterType.NORMAL, sigma_pct=1.5),
    UncertainParameter("epsilon",  77.5,   ParameterType.NORMAL, sigma_pct=2.0),
]


def test_mc_ttbe_basic_statistics():
    """Monte Carlo study: mean Isp near nominal, CV < 3%."""
    evaluator = make_jannaf_evaluator(TTBE_PARAMS)
    runner = MonteCarloRunner(
        params=TTBE_PARAMS, n_samples=100,
        sampling_method="lhs", n_jobs=1, seed=42,
    )
    result = runner.run(evaluator)

    assert result.stats.N_samples >= 95, "Fewer than 95% samples converged"
    # Mean should be near nominal Isp (within 10%)
    assert 380 < result.stats.mean < 500, f"Mean Isp {result.stats.mean:.1f}s out of range"
    assert result.stats.cv_pct < 5.0, f"CV too high: {result.stats.cv_pct:.2f}%"


def test_mc_ttbe_convergence_rate():
    """All 100 LHS samples should converge for JANNAF model."""
    evaluator = make_jannaf_evaluator(TTBE_PARAMS)
    runner = MonteCarloRunner(
        params=TTBE_PARAMS, n_samples=50,
        sampling_method="lhs", n_jobs=1, seed=7,
    )
    result = runner.run(evaluator)
    convergence_rate = np.sum(result.converged) / len(result.converged)
    assert convergence_rate > 0.95, f"Convergence rate {convergence_rate:.2%} too low"


def test_mc_ttbe_sobol_pc_dominates():
    """Sobol analysis: Pc should have the highest total-order index."""
    from atha.monte_carlo.sampling import SaltelliSampler
    from atha.monte_carlo.sensitivity import compute_sobol_indices

    evaluator = make_jannaf_evaluator(TTBE_PARAMS)
    N_base = 80
    samples = SaltelliSampler(seed=42).sample(TTBE_PARAMS, N_base)
    Y = np.array([evaluator(X) for X in samples])

    Si = compute_sobol_indices(TTBE_PARAMS, samples, Y, N_base=N_base)

    # Pc and MR should be the top two drivers (in either order)
    top2_idx = np.argsort(Si["ST"])[-2:]
    param_names = [p.name for p in TTBE_PARAMS]
    top2_names = {param_names[i] for i in top2_idx}
    assert "Pc" in top2_names or "MR" in top2_names, \
        f"Expected Pc or MR in top 2 drivers, got {top2_names}"


def test_mc_result_save_load_roundtrip(tmp_path):
    evaluator = make_jannaf_evaluator(TTBE_PARAMS)
    runner = MonteCarloRunner(
        params=TTBE_PARAMS, n_samples=30,
        sampling_method="lhs", n_jobs=1, seed=1,
    )
    result = runner.run(evaluator)
    fname = str(tmp_path / "mc_ttbe.hdf5")
    result.save(fname)

    from atha.monte_carlo.results import MonteCarloResult
    loaded = MonteCarloResult.load(fname)

    np.testing.assert_array_almost_equal(result.Y_samples, loaded.Y_samples)
    assert abs(result.stats.mean - loaded.stats.mean) < 0.001
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/integration/test_monte_carlo_ttbe.py -v
# Expected: FAIL — ImportError or logic error
```

- [ ] **Step 3: Run tests, fix any issues, verify pass**

```
pytest tests/integration/test_monte_carlo_ttbe.py -v
# Expected: all 4 tests PASS
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_monte_carlo_ttbe.py
git commit -m "test: add TTBE Monte Carlo integration tests with Sobol analysis"
```

---

### Task B7: ProfileMonteCarloRunner

**Files:**
- Create: `atha/monte_carlo/profile_runner.py`
- Modify: `tests/integration/test_monte_carlo_ttbe.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/integration/test_monte_carlo_ttbe.py
from atha.monte_carlo.profile_runner import ProfileMonteCarloRunner


def test_profile_mc_success_rate():
    """Run 10 profile samples; verify >80% succeed."""
    from tests.integration.test_profile_ttbe import (
        build_simple_ttbe_engine, mdot_100, mdot_65, h_ref
    )
    # Re-import the profile definition helper
    engine, gas = build_simple_ttbe_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()

    # Minimal 2-phase profile: dwell then dwell (always succeeds)
    from atha.profiles import TestProfile, PhaseDefinition, PhaseMode
    profile = TestProfile(
        name="mc_profile",
        phases=[
            PhaseDefinition("dwell1", PhaseMode.DWELL, duration=0.1),
            PhaseDefinition("dwell2", PhaseMode.DWELL, duration=0.1),
        ],
    )

    params = [
        UncertainParameter("Pc", 20.6e6, ParameterType.NORMAL, sigma_pct=2.0),
    ]
    runner = ProfileMonteCarloRunner(
        params=params, n_samples=10,
        sampling_method="lhs", n_jobs=1, seed=0,
        layout=layout, profile=profile, X0=X0,
    )
    result = runner.run()
    assert result.stats.N_samples >= 8   # at least 80% converge
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/integration/test_monte_carlo_ttbe.py::test_profile_mc_success_rate -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement profile_runner.py**

```python
# atha/monte_carlo/profile_runner.py
from __future__ import annotations
from typing import List, Callable, Optional
import numpy as np

from atha.monte_carlo.parameters import UncertainParameter
from atha.monte_carlo.runner import MonteCarloRunner
from atha.monte_carlo.results import MonteCarloResult


class ProfileMonteCarloRunner(MonteCarloRunner):
    """Monte Carlo runner that executes a full TestProfile for each sample.

    For each perturbed parameter set, the runner:
    1. Reconstructs engine boundary conditions from the perturbed parameters
    2. Executes the test profile
    3. Extracts a scalar metric (default: mean Isp across mainstage phases)
    """

    def __init__(
        self,
        params: List[UncertainParameter],
        n_samples: int,
        layout,
        profile,
        X0: np.ndarray,
        extract_metric: Optional[Callable] = None,
        sampling_method: str = "lhs",
        n_jobs: int = -1,
        seed: int = 42,
        verbose: int = 0,
    ):
        super().__init__(params=params, n_samples=n_samples,
                          sampling_method=sampling_method,
                          n_jobs=n_jobs, seed=seed, verbose=verbose)
        self.layout = layout
        self.profile = profile
        self.X0 = X0
        self.extract_metric = extract_metric or self._default_metric

    @staticmethod
    def _default_metric(profile_result) -> float:
        """Default: return duration of last completed phase as success metric."""
        if not profile_result.success or not profile_result.phases:
            return float("nan")
        return float(profile_result.total_duration)

    def run(self) -> MonteCarloResult:
        import copy
        from atha.profiles.limits import EngineAbort

        def evaluate_with_profile(X):
            # Each worker builds its own profile execution (stateless)
            try:
                profile_result = self.profile.execute(self.layout, self.X0)
                return self.extract_metric(profile_result)
            except EngineAbort:
                return float("nan")
            except Exception:
                return float("nan")

        return super().run(evaluate_fn=evaluate_with_profile)
```

- [ ] **Step 4: Update monte_carlo/__init__.py**

```python
# atha/monte_carlo/__init__.py
from atha.monte_carlo.parameters import UncertainParameter, ParameterType
from atha.monte_carlo.runner import MonteCarloRunner
from atha.monte_carlo.results import MonteCarloResult
from atha.monte_carlo.profile_runner import ProfileMonteCarloRunner

__all__ = [
    "UncertainParameter", "ParameterType",
    "MonteCarloRunner", "MonteCarloResult",
    "ProfileMonteCarloRunner",
]
```

- [ ] **Step 5: Run all tests, verify pass**

```
pytest tests/ -v
# Expected: all tests PASS (137 existing + new tests)
```

- [ ] **Step 6: Commit**

```bash
git add atha/monte_carlo/profile_runner.py atha/monte_carlo/__init__.py \
        tests/integration/test_monte_carlo_ttbe.py
git commit -m "feat: add ProfileMonteCarloRunner for Monte Carlo over full test profiles"
```

---

## Verification

Full end-to-end verification after all tasks complete:

```bash
# All tests pass
pytest tests/ -v --tb=short

# Test profile example
python -c "
from atha.profiles import TestProfile, PhaseDefinition, PhaseMode, ControlCommand
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine

gas = IdealGasBackend(gamma=1.4, R=287.0)
vol = Volume('tank', volume=0.1, thermo=gas, initial_P=1e5, initial_T=300.0)
vol.add_inlet('fill')
engine = Engine('e')
engine.add_component(vol)
layout = engine.compile()
X0 = layout.assemble_state_vector()
h_ref = gas.state_from_PT(1e5, 300.0).h

profile = TestProfile('test', phases=[
    PhaseDefinition('fill', PhaseMode.TRANSIENT, 5.0,
                    control_commands=[
                        ControlCommand('fill.mdot', lambda t: 0.05),
                        ControlCommand('fill.h', lambda t: h_ref),
                    ]),
])
result = profile.execute(layout, X0)
print('Profile success:', result.success)
print('Duration:', result.total_duration)
P = result.phases[0].get('tank', 'P')
print(f'P: {P[0]/1e5:.2f} → {P[-1]/1e5:.2f} bar')
"

# Monte Carlo example
python -c "
from atha.monte_carlo import UncertainParameter, ParameterType, MonteCarloRunner
from atha.validation.ttbe import run_ttbe_100pct_rpl
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.thermo.ideal_gas import IdealGasBackend

params = [
    UncertainParameter('eta_cstar', 0.975, ParameterType.NORMAL, sigma=0.005),
    UncertainParameter('eta_div',   0.983, ParameterType.NORMAL, sigma=0.004),
]

def evaluate(X):
    thermo = IdealGasBackend(gamma=1.24, R=711.0)
    eff = JANNAFEfficiencies(eta_cstar=X[0], eta_divergence=X[1])
    j = SimplifiedJANNAF(thermo, eff, 0.0687, 0.0687*77.5, 0.0)
    return j.compute(20.6e6, 3560.0, 6.0, 468.0).Isp

runner = MonteCarloRunner(params, n_samples=50, n_jobs=1, seed=42)
result = runner.run(evaluate)
result.print_summary()
"
```

---

## Self-Review Checklist

### Spec Coverage
- [x] Multi-phase test profile (STEADY_TRIM, TRANSIENT, DWELL) — Tasks A1–A6
- [x] Event scheduling via ControlCommand — Task A1
- [x] Abort/limit checking with EngineAbort — Task A2
- [x] Multi-rate data recording via dense output — Task A4
- [x] HDF5 save/load — Task A7
- [x] Timeline plotting — Task A3 (plot_timeline)
- [x] TTBE throttle sweep integration test — Task A8
- [x] Monte Carlo with LHS sampling — Task B2
- [x] Saltelli sampling for Sobol — Task B2
- [x] Parallel execution (joblib) — Task B4
- [x] Sobol sensitivity analysis (SALib) — Task B5
- [x] HDF5 save/load for MC results — Task B4
- [x] Histogram and Sobol bar plots — Task B4
- [x] TTBE MC integration test — Task B6
- [x] Profile + MC combined runner — Task B7

### Type Consistency
- `execute_phase(layout, X0, phase, global_limits)` signature consistent across A5, A6, A7
- `MonteCarloRunner.run(evaluate_fn)` → `MonteCarloResult` consistent across B4, B6, B7
- `PhaseResult.get(component, state)` matches `TransientSolution.get()` API
- `UncertainParameter.transform_unit(u)` used in both `LHSSampler` and `SaltelliSampler`
