# Test Profile Research: Engine Simulation Architecture

## Executive Summary

This document synthesizes best practices from ROCKETS (NASA/P&W 1991), NPSS, ESPSS, and GSP for implementing engine test profile simulation in ATHA. The key insight is that test profiles are **multi-phase control sequences** combining steady-state trim points, transient dynamics, and event-driven transitions. ATHA's existing architecture (Radau ODE solver + Newton DAE) is well-suited to implement this with minimal extensions.

---

## Part 1: Current ATHA Capabilities Review

### Current Strengths

1. **Compilation step** (`Engine.compile()`) produces `EngineLayout` with fixed evaluation order — pre-allocated arrays enable efficient batch evaluation, essential for test profiles executing many sequential trim points.

2. **Radau ODE solver** with adaptive step size — A-stable, handles variable step sizes needed for multi-phase sequences. Dense output allows fixed-rate data recording without affecting step selection.

3. **Newton-Raphson steady-state** — existing `SteadyStateSolver` closes all algebraic loops simultaneously. Suitable for trim points within a profile.

4. **State handoff** — `assemble_state_vector()` and `scatter_state_vector()` provide clean mechanism to pass end-state of one phase as initial condition of next.

### Current Gaps

1. **No test phase abstraction**: `TransientSolver.integrate()` covers a single t_span interval. No concept of a named phase.
2. **No event scheduling**: No time-triggered or condition-triggered events.
3. **No automatic steady-state detection**: Manual t_span specification required.
4. **No abort/limit checking**: No safety interlock framework.
5. **No multi-trim sequencing**: No way to specify "start at trim A, transient to trim B."
6. **No multi-rate recording**: Data only available at integrator step times (non-uniform).

---

## Part 2: How ROCKETS Handles Multi-Phase Simulation

### Three-Run Mode Design (ROCKETS Section 2.1)

ROCKETS solves engine test profiles as an explicit state machine with three alternating modes:

#### Mode 1: Steady-State Trim Balance
```
Objective: Drive all dX/dt = 0 simultaneously
Solver:    Modified Newton-Raphson on residual vector F(X) = dX/dt
Closes all algebraic loops (e.g., pressures, flow splits, power balance)
Validates:
  - Turbopump power balance
  - Pressure drops match flow rates
  - Combustor mixture ratio consistent with injector characteristics
```

**Key ROCKETS insight**: Trim is not "wait long enough for transients to settle" but an **algebraic constraint problem**. States that violate physical constraints (e.g., MR out of bounds) fail to converge — this is deliberate, not a bug.

#### Mode 2: Transient Integration
```
Duration: ΔT seconds or until next scheduled event
Integration: Gear/trapezoidal (Radau equivalent)
Outputs: Time series of all states
Corrector equations solved by modified Newton-Raphson at each step
States can be "pinned" (held at steady-state value) or "active" (free to change)
```

**Key insight**: Individual states can be selectively activated/deactivated. Useful for engine startup where turbine speed is pinned until sufficient flow develops, or for creating reduced-order models.

#### Mode 3: Linearization (Optional)
```
Around trim point: Compute state-space matrices A, B, C, D
Used for: Control law development, frequency response analysis
Not needed for basic test profiles but useful for controller design validation
```

### ROCKETS Test Profile Structure (Implicit in Section 5 TTBE Validation)

ROCKETS profiles specify:
1. **Initial steady-state trim** (Pc_target, MR_target, throttle level)
2. **Phase list**: sequence of events/durations
3. **Per-phase parameters**: ramp rates, hold times, target values
4. **Abort criteria**: limit checks (T_wall < 400K, Pc > 25 MPa, ω < ω_min)

Example from TTBE validation (100% → 65% → 100% RPL throttle sweep):
```
Phase 1: Steady-state at 100% RPL (Pc = 20.6 MPa)
  - Target: Pc=20.6e6 Pa, MR=6.0, throttle=1.0
  
Phase 2: Transient ramp to 65% RPL
  - Duration: 5 seconds
  - Throttle command: 1.0 → 0.65 (linear ramp)
  - Abort if: |dPc/dt| > 1 MPa/s (overshoot protection)
  
Phase 3: Hold at 65% RPL (steady-state trim)
  
Phase 4: Ramp back to 100% RPL
  - Duration: 5 seconds
  - Throttle command: 0.65 → 1.0
  
Phase 5: Shutdown
  - Close all propellant valves
  - Monitor pressure decay
```

---

## Part 3: Event Scheduling Patterns

### Time-Triggered Events

The simplest and most reliable form. `scipy.integrate.solve_ivp` supports events via zero-crossing callbacks:

```python
def valve_open_event(t, X):
    return t - 2.5   # zero at t=2.5, triggers event

valve_open_event.terminal = True   # stop integration at event
valve_open_event.direction = 1     # trigger on upward zero-crossing

sol = solve_ivp(rhs, t_span, X0, events=[valve_open_event])
# sol.t_events[0] contains the exact crossing time
```

**Pattern for ATHA**: Wrap all time events in a common abstraction:
```python
@dataclass
class TimeEvent:
    trigger_time: float
    description: str
    terminal: bool = True   # terminal=True → stop integration, re-start next phase

    def as_scipy_event(self):
        def fn(t, X):
            return t - self.trigger_time
        fn.terminal = self.terminal
        fn.direction = 1
        return fn
```

### Condition-Triggered Events (Limit Events)

```python
@dataclass
class LimitEvent:
    state_key: str          # "chamber.P" or component+state descriptor
    upper_limit: float      # None to skip upper check
    lower_limit: float      # None to skip lower check
    is_hard: bool           # True=abort, False=log-only
    description: str

    def as_scipy_event(self):
        def fn(t, X):
            # Returns negative if inside limit, positive if exceeded
            value = extract_state_value(X, self.state_key)
            if self.upper_limit is not None:
                return self.upper_limit - value   # zero when value == upper_limit
        fn.terminal = self.is_hard
        fn.direction = -1   # trigger when value crosses limit going up
        return fn
```

### Hysteresis for Noisy Signals

For pressure oscillations and sensor noise, simple limit checks produce spurious triggers. Use hysteresis:

```python
@dataclass
class HysteresisLimit:
    state_key: str
    upper_arm: float    # trigger on rising at this value
    lower_arm: float    # reset on falling at this value
    _triggered: bool = False

    def check(self, value):
        if not self._triggered and value > self.upper_arm:
            self._triggered = True
            return True    # new trigger
        if self._triggered and value < self.lower_arm:
            self._triggered = False
        return False
```

---

## Part 4: NPSS and GSP Test Sequence Architecture

### NPSS Approach (NASA Glenn → SwRI)

**NPSS philosophy**: "Numerical Test Cell" — component-based ODE/DAE framework, same philosophy as ATHA.

Key NPSS features relevant to test profiles:
- States have an explicit `"active"` flag (dynamic) or `"pinned"` (algebraic) — allows selective pinning during startup
- Custom first-order lags can be added to any component output (e.g., actuator dynamics)
- Adaptive time-stepping per component based on local time constants
- Control system integration via direct coupling to MATLAB/Simulink

**NPSS test profiles** are defined as control laws + setpoint ramps in a sequence file:
```
# Example NPSS-style profile (pseudo-syntax)
t=0:   Pc_setpoint = 20.6e6 Pa
t=5:   ramp(Throttle, from=1.0, to=0.65, duration=5s, shape=linear)
t=10:  Pc_setpoint = 13.4e6 Pa  (65% RPL)
t=15:  ramp(Throttle, from=0.65, to=1.0, duration=5s)
t=20:  Pc_setpoint = 20.6e6 Pa  (100% RPL)
```

### GSP Approach (NLR — Gas turbine Simulation Program)

GSP's explicit phase structure closely matches what ATHA needs:

```python
# GSP phase concept (adapted)
class Phase:
    name: str                    # "startup", "mainstage", "throttledown"
    duration: float              # seconds or "until_convergence"
    solver_mode: str             # "steady" or "transient"
    trim_targets: Dict           # {Pc: 20.6e6, MR: 6.0} for steady phases
    control_laws: List           # Callable ramp functions for transient phases
    initial_conditions: Dict     # from previous phase (handed off automatically)
```

**Key GSP insight**: Each phase explicitly specifies solver mode:
- **Startup → transient** (captures fuel ramp, ignition dynamics)
- **Mainstage → steady-state trim** (only interested in equilibrium performance)
- **Throttle sweep → transient** (captures rotor acceleration, pressure lag)

### ESPSS Approach (ESA/EcosimPro)

ESPSS uses "Scenario" files for multi-phase sequences:
```
Scenario: RL-10A-3-3A Ignition and Mainstage

[Phase: Pressurize]
  Duration: 0.5 s
  Tank_Valve: 0 → 0.5 (ramp)
  Abort_If: Pc > 1 MPa

[Phase: Ignition]
  Duration: 1.0 s
  Igniter: On
  Main_Valve: 0.5 → 1.0 (ramp)
  Abort_If: Pc > 5 MPa

[Phase: Mainstage]
  Duration: 60.0 s

[Phase: Shutdown]
  Duration: 5.0 s
  Main_Valve: 1.0 → 0 (ramp)
```

---

## Part 5: Data Recording and Sampling Strategy

### Multi-Rate Data Acquisition (Industry Practice)

Real engine test cells record at multiple sampling rates:

| Parameter | Rate | Rationale |
|-----------|------|-----------|
| Pressures, temperatures | 10 Hz | Thermal time constants ~0.1 s |
| Rotor speeds, valve positions | 100 Hz | Control bandwidth ~10 Hz |
| Accelerometer data | 10 kHz | Structural modes |
| Throat heat flux sensors | 1 kHz | Sensor bandwidth limit |

### Recording via Dense Output

Radau's `dense_output=True` provides an interpolating polynomial between steps. This decouples data recording from step selection:

```python
# After solve_ivp with dense_output=True:
sol.sol              # Dense OdeSolution object

# Uniform-time recording at any rate, without recomputing:
t_record = np.arange(t0, tf, 1/100.0)   # 100 Hz
X_record = sol.sol(t_record)             # shape (n_states, N_record)
```

**Key advantage**: Setting recording_rate = 100 Hz does not force the integrator to take 100 Hz steps. Radau takes steps sized by error control (often 10–50 ms), and dense output interpolates between them.

### Abort Data Dumps

When a limit event triggers or an abort occurs:
```python
# Save the last 10 seconds of data before abort
pre_abort_mask = sol.t >= (t_abort - 10.0)
abort_dump = {
    "time": sol.t[pre_abort_mask],
    "states": sol.y[:, pre_abort_mask],
    "abort_reason": str(reason),
    "abort_time": t_abort,
}
```

---

## Part 6: Recommended ATHA Test Profile Architecture

### Core Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
from enum import Enum

class PhaseMode(Enum):
    STEADY_TRIM = "steady_trim"     # Newton solve dX/dt=0
    TRANSIENT = "transient"         # Radau ODE integration
    DWELL = "dwell"                 # Hold current state (no solve)

@dataclass
class ControlCommand:
    """A time-varying command applied to a BCS key."""
    bcs_key: str                    # e.g., "throttle.cmd"
    fn: Callable[[float], float]    # fn(t_phase) → value
    # t_phase is time since start of this phase (0-based)

@dataclass
class SafetyLimit:
    name: str
    component_name: str             # e.g., "chamber"
    state_name: str                 # e.g., "P"
    upper_limit: Optional[float] = None
    lower_limit: Optional[float] = None
    is_hard: bool = True            # abort vs. warn-only
    description: str = ""

@dataclass
class PhaseDefinition:
    name: str
    mode: PhaseMode
    duration: float                 # seconds; for STEADY_TRIM: max solver time
    trim_targets: Dict[str, float] = field(default_factory=dict)
    control_commands: List[ControlCommand] = field(default_factory=list)
    abort_checks: List[SafetyLimit] = field(default_factory=list)
    recording_rate_hz: float = 100.0
    solver_options: Dict = field(default_factory=dict)  # rtol, atol, max_step overrides

@dataclass
class TestProfile:
    name: str
    phases: List[PhaseDefinition]
    global_limits: List[SafetyLimit] = field(default_factory=list)

    def execute(self, layout, X0) -> "TestProfileResult": ...
```

### Phase Execution Logic

```python
def execute_phase(layout, X0, phase: PhaseDefinition,
                  all_limits: List[SafetyLimit]) -> "PhaseResult":

    if phase.mode == PhaseMode.STEADY_TRIM:
        solver = SteadyStateSolver(layout)
        X_final = solver.solve(X0, phase.trim_targets)
        return PhaseResult(name=phase.name, X_final=X_final,
                           t=np.array([0.0, phase.duration]),
                           X=np.vstack([X0, X_final]))

    elif phase.mode == PhaseMode.TRANSIENT:
        # Build scipy event callbacks
        scipy_events = []
        for limit in all_limits:
            scipy_events.append(limit_to_scipy_event(layout, limit))

        # Build BCS callable from control commands
        def bcs(t_abs):
            t_phase = t_abs  # t is already phase-local (solver restarts at 0)
            result = {}
            for cmd in phase.control_commands:
                result[cmd.bcs_key] = cmd.fn(t_phase)
            return result

        solver = TransientSolver(layout,
                                  rtol=phase.solver_options.get("rtol", 1e-4),
                                  atol=phase.solver_options.get("atol", 1e-6),
                                  max_step=phase.solver_options.get("max_step", 0.01))
        sol = solver.integrate((0.0, phase.duration), X0, bcs,
                                events=scipy_events)

        # Downsample to recording rate
        t_rec = np.arange(0, phase.duration, 1.0/phase.recording_rate_hz)
        X_rec = sol.sol(t_rec).T if sol.sol else sol.X   # interpolate via dense output

        return PhaseResult(name=phase.name, X_final=sol.X[-1],
                           t=t_rec, X=X_rec,
                           abort_triggered=len(sol.t_events[0]) > 0)
```

---

## Part 7: Example TTBE Throttle Sweep Profile

```python
from atha.profiles import TestProfile, PhaseDefinition, PhaseMode
from atha.profiles import ControlCommand, SafetyLimit
import numpy as np

ttbe_profile = TestProfile(
    name="TTBE_100_65_100",
    phases=[
        # Phase 1: trim at 100% RPL
        PhaseDefinition(
            name="mainstage_100pct",
            mode=PhaseMode.STEADY_TRIM,
            duration=30.0,
            trim_targets={"chamber.P": 20.6e6, "MR": 6.0},
            recording_rate_hz=10.0,
        ),

        # Phase 2: throttle down to 65%
        PhaseDefinition(
            name="throttle_down",
            mode=PhaseMode.TRANSIENT,
            duration=5.0,
            control_commands=[
                ControlCommand("throttle.cmd",
                               fn=lambda t: 1.0 - 0.35 * min(t / 5.0, 1.0)),
            ],
            abort_checks=[
                SafetyLimit("Pc_overshoot", "chamber", "P",
                            upper_limit=22e6, is_hard=True),
            ],
            recording_rate_hz=100.0,
        ),

        # Phase 3: trim at 65% RPL
        PhaseDefinition(
            name="mainstage_65pct",
            mode=PhaseMode.STEADY_TRIM,
            duration=30.0,
            trim_targets={"chamber.P": 13.4e6, "MR": 6.0},
            recording_rate_hz=10.0,
        ),

        # Phase 4: throttle up to 100%
        PhaseDefinition(
            name="throttle_up",
            mode=PhaseMode.TRANSIENT,
            duration=5.0,
            control_commands=[
                ControlCommand("throttle.cmd",
                               fn=lambda t: 0.65 + 0.35 * min(t / 5.0, 1.0)),
            ],
            recording_rate_hz=100.0,
        ),

        # Phase 5: final 100% RPL trim
        PhaseDefinition(
            name="mainstage_100pct_final",
            mode=PhaseMode.STEADY_TRIM,
            duration=30.0,
            trim_targets={"chamber.P": 20.6e6, "MR": 6.0},
            recording_rate_hz=10.0,
        ),
    ],
    global_limits=[
        SafetyLimit("Pc_hard_max", "chamber", "P",
                    upper_limit=25e6, is_hard=True,
                    description="Absolute chamber pressure limit"),
        SafetyLimit("MR_range", "chamber", "P",
                    lower_limit=4.0, upper_limit=8.0, is_hard=False,
                    description="Mixture ratio soft limit"),
    ],
)

# Execute
result = ttbe_profile.execute(layout=ttbe_layout, X0=X0_cold)
result.save("ttbe_throttle_sweep.hdf5")
result.plot_timeline()   # Pc, Isp, throttle vs. time, phase markers
```

---

## References

1. **ROCKETS System**: NASA/P&W, November 1991 (19910011919.pdf) — Sections 2, 5
2. **NPSS Documentation**: NASA Glenn Research Center
3. **GSP Manual**: NLR Gas turbine Simulation Program
4. **ESPSS Toolkit**: EcosimPro-based ESA propulsion simulation
5. **Overview of Rocket Engine Control**: NASA TM 4055 (1992)
6. **ATHA Architecture Decisions**: `05_architecture_decisions.md`
7. **ATHA Numerical Methods**: `07_numerical_methods.md`
