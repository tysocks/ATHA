# ATHA YAML Configuration Implementation

Date: 2026-05-07

## Purpose

ATHA uses a modular YAML configuration system with one top-level Analysis YAML
as the runnable entrypoint. The engine topology remains stable while maps,
transient behavior, test boundaries, operating targets, timings, controllers,
telemetry, solver options, and analysis settings can be replaced independently.

This mirrors the useful separation in ROCETS: configuration, run setup,
execution, and output definition are separate concerns, but a single run
manifest can bind them into one simulation.

## File Responsibilities

### Analysis YAML

The Analysis YAML is the only required entrypoint for a complete run. It owns:

- engine YAML reference
- map slot bindings
- transient slot bindings
- boundary-condition file reference
- operating-condition file reference
- timing file reference
- controller file reference
- telemetry file reference
- solver options
- nominal, sweep, or Monte Carlo analysis settings

Example:

```yaml
name: gg_start_transient
engine: engines/gg_engine.yaml

maps:
  lox_pump_combo: maps/lox_pump_combo.yaml
  main_lox_valve_cda: maps/main_lox_valve_cda.yaml

transients:
  main_lox_valve: transients/main_lox_valve.yaml

boundary_conditions: profiles/startup_boundaries.yaml
operating_conditions: profiles/startup_targets.yaml
timings: profiles/startup_timings.yaml
controllers: profiles/startup_controllers.yaml
telemetry: profiles/startup_telemetry.yaml

solver:
  mode: transient
  transient:
    method: Radau
    rtol: 1.0e-4
    atol: 1.0e-6
    max_step: 0.002
    split_at_timing_events: true
  steady_trim:
    tol: 1.0e-8
    max_iter: 200

analysis:
  type: nominal
  output_dir: outputs/gg_start_transient
```

The default solver settings should track the currently successful implementation:
Radau transient integration, `rtol=1.0e-4`, `atol=1.0e-6`, and phase-level
`max_step` control.

### Engine YAML

The Engine YAML defines stable hardware structure:

- components
- connections/layout
- fixed component parameters
- map slots used by components
- transient slots used by components
- initial state guesses where needed

It should not embed large map tables or full transient calibrations.

```yaml
name: gg_engine
units: SI

components:
  lox_pump:
    type: Pump
    parameters:
      diameter: 0.12
      mdot_design: 12.4
      omega_design: 3141.59
    maps:
      head_map:
        ref: lox_pump_combo
        output: head
      efficiency_map:
        ref: lox_pump_combo
        output: efficiency

  main_lox_valve:
    type: Valve
    parameters:
      max_area: 3.1e-4
      discharge_coeff: 0.72
    maps:
      cda_map: main_lox_valve_cda
    transient: main_lox_valve

connections:
  - from: main_lox_valve.outlet
    to: lox_pump.inlet
    domain: fluid
```

### Map YAML

Map YAML defines where map data comes from and how to interpret it. Large data
lives in CSV/HDF5 exported from tests, component sweeps, or external simulation.
Small constants may be embedded.

Separate map files are preferred when relations are calibrated independently.
Multi-output maps are preferred when several outputs share the same axes and
come from the same test or simulation sweep.

```yaml
name: lox_pump_combo
kind: structured_grid
source:
  type: csv
  path: data/lox_pump_combo.csv

axes:
  - name: corrected_speed
    column: Nc
  - name: corrected_flow
    column: Wc

outputs:
  - name: head
    column: head_j_per_kg
  - name: efficiency
    column: eta

interpolation:
  method: linear
  extrapolation: clamp
```

For a component with multiple maps, the Engine YAML binds each component slot:

```yaml
maps:
  head_map: lox_pump_head
  efficiency_map: lox_pump_efficiency
```

or binds different outputs from one map:

```yaml
maps:
  head_map:
    ref: lox_pump_combo
    output: head
  efficiency_map:
    ref: lox_pump_combo
    output: efficiency
```

### Transient YAML

Transient YAML defines reusable dynamic behavior for actuator-like component
parameters.

```yaml
name: main_lox_valve
type: first_order_rate_limited
state:
  name: position
  initial: 0.0
  lower: 0.0
  upper: 1.0
command:
  name: command.position
parameters:
  time_constant: 0.08
  opening_rate_limit: 3.0
  closing_rate_limit: 5.0
  deadband: 0.002
```

### Boundary Conditions YAML

Boundary conditions are imposed physical environment or test-stand values. They
may vary with time.

```yaml
name: startup_boundaries
time_unit: s
conditions:
  lox_tank.outlet.P:
    units: Pa
    schedule:
      type: table
      values:
        - [0.0, 5.2e6]
        - [2.0, 5.1e6]
        - [8.0, 4.8e6]
  nozzle.ambient.P:
    units: Pa
    value: 101325.0
```

### Operating Conditions YAML

Operating conditions are targets consumed by controllers. They do not directly
overwrite physical states.

```yaml
name: startup_targets
targets:
  chamber_pressure:
    variable: chamber.P
    units: Pa
    schedule:
      type: ramp
      t_start: 1.0
      t_end: 4.0
      y_start: 1.0e6
      y_end: 7.0e6
```

### Timing YAML

Timings are direct event or command schedules, such as valve opens, ignition
enable, purge start, and shutdown. Transient integration should split at known
event times.

```yaml
name: startup_timings
events:
  - name: main_valves_open
    time: 1.0
    commands:
      main_lox_valve.command.position:
        type: ramp
        duration: 0.35
        final: 1.0
```

### Controllers YAML

Controllers convert operating targets into actuator commands.

```yaml
name: startup_controllers
controllers:
  pc_controller:
    type: PID
    sensor:
      variable: chamber.P
    setpoint: targets.chamber_pressure
    output:
      command: main_fuel_valve.command.position
      lower: 0.0
      upper: 1.0
    gains:
      kp: 2.0e-7
      ki: 1.5e-6
      kd: 0.0
    anti_windup: clamp
```

### Telemetry YAML

Telemetry defines what to export and the alias/channel name to use.

```yaml
name: startup_telemetry
sample_rate_hz: 1000
channels:
  - alias: PC_CHAMBER
    source: chamber.P
    units: MPa
  - alias: LOX_VALVE_POS
    source: main_lox_valve.position
exports:
  csv: true
  hdf5: true
```

## Implemented Foundation

- `atha.config.schema` defines typed dataclass schemas for Analysis, Engine,
  Map, Transient, Boundary Conditions, Operating Conditions, Timings,
  Controllers, and Telemetry YAML files.
- `atha.config.loader.load_analysis_config()` loads one Analysis YAML and all
  referenced files, resolving relative paths from the file that contains the
  reference.
- Engine component map bindings support both simple references and multi-output
  bindings with explicit output selection.
- Loader validation catches missing map/transient bindings and missing requested
  map outputs before a simulation starts.
- `atha.config.maps.build_performance_map()` turns Map YAML into runtime
  `PerformanceMap` instances for CSV, HDF5, and constant sources.
- `atha.config.schedules` evaluates constant, table, ramp, and step schedules
  for boundary conditions and operating targets.

## Next Implementation Steps

- Add an engine builder that constructs `Engine` objects from `EngineConfig`
  using the existing component constructors.
- Add schedule evaluators for boundary conditions, operating targets, and timing
  command ramps/tables.
- Add transient component state support, starting with valve actual position.
- Add controller runtime objects and profile integration.
- Extend `TestProfileResult` to record telemetry channels, algebraics, commands,
  controller signals, outputs, and residual diagnostics.
- Add Analysis YAML execution for nominal, structured sweep, and Monte Carlo
  modes.
