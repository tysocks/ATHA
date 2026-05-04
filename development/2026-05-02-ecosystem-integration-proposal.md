# ATHA Ecosystem Integration — Proposed Changes
## Status: PROPOSED — deferred pending ecosystem stabilization
## Date: 2026-05-02

---

## Context

ATHA is one of four projects in the Northern Stream ecosystem:

| Project | Role |
|---------|------|
| **NORDSTROM** | Central database — RedscaleDB (real test data) + BluescaleDB (simulation data) |
| **LabVIEW** | Real-time test stand controller — DAQ, valve sequencing, abort monitoring, TDMS logging |
| **NOVA** | Data viewer — reads both databases, overlays simulation vs. test time series |
| **ATHA** | Simulation tool — pre-test performance prediction and uncertainty quantification |

The integration goal is: simulate a test profile in ATHA, run that exact same profile on the test stand via LabVIEW, ingest the real data into NORDSTROM, and compare simulation vs. reality side-by-side in NOVA.

---

## Proposed Integration Architecture

### Operational Mode: Offline Pre-Test Prediction (Option A)

ATHA runs offline before test day. It:
1. Reads a shared profile JSON
2. Simulates the full test with an embedded closed-loop controller
3. Predicts all channel time histories
4. Flags any abort limit violations before hardware is touched
5. Writes the prediction to NORDSTROM BluescaleDB
6. NOVA overlays prediction vs. post-test reality

No real-time coupling between ATHA and LabVIEW. LabVIEW runs independently on test day.

---

## Proposed Design Decisions

### Decision 1: Controller Architecture — PID as ATHA Component

A `PIDController` extends `BaseComponent` and wires into the engine graph. It reads a measurement state from another component and outputs an actuator command into BCS. The controller dynamics run inside the ODE solver's time loop (Radau), giving accurate simulation of fast pressure dynamics. Anti-windup included.

**Rejected alternatives:**
- Controller at phase executor level (breaks Radau adaptive stepping, loses accuracy on fast dynamics)
- Pre-computed feedforward lookup table (no feedback, misses transients)

### Decision 2: Shared Profile JSON — Setpoint-Centric with Embedded Controller Spec

One JSON file encodes both the test intent (setpoints as waypoints) and the control architecture (PID gains, actuator mapping). LabVIEW reads only the setpoint tables and abort limits. ATHA reads the full spec.

**Rationale:** Single source of truth for a test campaign. No sync problems between separate files.

---

## Proposed Shared Profile JSON Schema

Location: a shared `profiles/` folder outside both projects (tracked alongside NORDSTROM or in a shared config repo).

```json
{
  "schema_version": "1.0",
  "profile_id": "HFR-NOM-001",
  "test_type": "HFR",
  "description": "Nominal 15s hotfire at 2 MPa Pc",
  "engine_model": "prometheus_dev1",
  "phases": [
    {
      "name": "mainstage",
      "mode": "transient",
      "duration_s": 15.0,
      "recording_rate_hz": 100,
      "setpoints": [
        {
          "channel": "chamber_pressure",
          "unit": "Pa",
          "waypoints": [[0, 1.0e6], [2, 2.0e6], [8, 2.0e6], [13, 0.5e6]]
        }
      ],
      "controllers": [
        {
          "id": "Pc_ctrl",
          "type": "PID",
          "process_variable": "chamber_pressure",
          "manipulated_variable": "fuel_valve.position",
          "gains": { "Kp": 0.01, "Ki": 0.001, "Kd": 0.0 },
          "output_range": [0.0, 1.0],
          "sample_rate_hz": 100
        }
      ]
    }
  ],
  "global_abort_limits": [
    {
      "trigger_id": "ABT-001",
      "name": "Chamber Overpressure",
      "channel": "chamber_pressure",
      "condition": "greater_than",
      "threshold": 2.5e6,
      "is_hard": true
    }
  ],
  "channel_map": {
    "chamber_pressure":    "chamber.P",
    "thrust":              "nozzle.thrust",
    "fuel_valve_position": "fuel_valve.position",
    "chamber_temperature": "chamber.T"
  }
}
```

**LabVIEW reads:** `setpoints` (as its target profile), `global_abort_limits`
**ATHA reads:** everything
**NORDSTROM/NOVA:** use the flat channel names from `channel_map`

---

## New ATHA Modules Required

### 1. `atha/components/pid_controller.py`
- Extends `BaseComponent`
- States: `[error_integral]`
- Inputs: `setpoint`, `measurement`
- Output: `command` (clamped to `[output_min, output_max]`)
- Anti-windup via integrator clamping
- Discrete sample rate (hold output between samples)

### 2. `atha/components/throttle_valve.py`
- Extends `BaseComponent`
- State: `position` (0–1)
- ODE: `dposition/dt = (commanded - position) / tau` (first-order lag)
- Configurable slew rate limit
- Flow coefficient curve (Cv vs. position) for pressure drop calculation

### 3. `atha/components/solenoid_valve.py`
- Algebraic component (no ODE)
- Binary open/closed based on boolean command
- Optional: transition lag (finite open/close time)

### 4. `atha/profiles/schema.py` + `atha_profile_schema.json`
- JSON Schema definition for the shared format
- `jsonschema`-based validator
- Version field for forward compatibility

### 5. `atha/profiles/importer.py`
- Converts shared profile JSON → `TestProfile`
- Waypoints → `ControlCommand(fn=lambda t: np.interp(t, times, values))`
- Abort limits → `SafetyLimit` via `channel_map` reverse lookup
- Controller specs → `PIDController` components added to engine graph
- Returns `(TestProfile, modified_layout)`

### 6. `atha/integrations/nordstrom.py`
- Connects to NORDSTROM BluescaleDB (PostgreSQL)
- Writes `simulation_runs`, `simulation_channels`, `simulation_results`
- Channel names written in NORDSTROM flat convention (via `channel_map`)
- Config: host, port, db name, credentials (env vars or config file)

### 7. `atha/integrations/channel_map.py`
- Bidirectional mapping: ATHA dot notation ↔ NORDSTROM underscore names
- Loaded from the `channel_map` block in the profile JSON
- Used by NORDSTROM writer and abort limit importer

### 8. `atha/integrations/prediction.py`
- High-level API: `run_prediction(profile_path, layout, X0) -> PredictionResult`
- Chains: load JSON → validate → build profile → run simulation → write to NORDSTROM
- Returns prediction report with abort check summary and key metrics

---

## Unknowns That Must Be Resolved Before Implementation

1. **LabVIEW profile format is WIP.** The exact fields LabVIEW will read from the shared JSON are not finalized. The setpoint/waypoint structure proposed above may need to change to match LabVIEW's final implementation.

2. **Engine model mapping.** The `engine_model` field in the JSON needs to reference a specific ATHA engine definition. The mechanism for loading/registering named engine models is not yet designed.

3. **PID tuning methodology.** What process variables are actually controllable, what actuators exist, and what gain ranges are physically reasonable requires hardware characterization data not yet available.

4. **NORDSTROM BluescaleDB credentials and network access.** The simulation_runs/simulation_channels/simulation_results schema is defined but the write path from a simulation workstation to the database server needs to be tested.

5. **NOVA simulation display.** NOVA currently has no UI for overlaying a simulation_run against a test_run. This comparison view needs to be added to NOVA independently before the ATHA integration is useful end-to-end.

6. **Valve transient models.** The physical time constants (tau) and flow characteristics (Cv curves) for the actual hardware valves are not yet characterized.

---

## Recommended Implementation Order (when ready)

1. Finalize LabVIEW profile JSON schema (LabVIEW project)
2. Add simulation overlay to NOVA (NOVA project)
3. Implement `channel_map.py` + `schema.py` + `importer.py` (ATHA)
4. Implement `PIDController` + `ThrottleValve` components (ATHA)
5. Implement `nordstrom.py` writer (ATHA)
6. Implement `prediction.py` runner (ATHA)
7. End-to-end test: profile JSON → ATHA simulation → NORDSTROM → NOVA
