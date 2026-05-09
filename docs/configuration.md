# ATHA YAML Configuration Guide

ATHA runs from one top-level `analysis.yaml`. That file references the modular
files that describe the engine, maps, boundaries, operating targets,
controllers, timings, transients, telemetry, and solver settings.

## File Roles

| File | Purpose | Typical Contents |
| --- | --- | --- |
| `analysis.yaml` | Run manifest | file references, solver settings, analysis type, output names |
| `engine.yaml` | Physical model | components, component parameters, map slots, connections |
| `maps/*.yaml` | Performance data binding | CSV/HDF5/constant map source, axes, outputs |
| `boundaries.yaml` | Test stand/environment inputs | supply pressures, temperatures, densities, ambient pressure |
| `operating_conditions.yaml` | Controlled targets | thrust, chamber pressure, mass flow, OF, schedules/profiles |
| `controller.yaml` | Control laws | null, proportional, PI/PID, splitters, limiters, custom functions |
| `timings.yaml` | Direct commands/events | valve open/close commands and other scripted events |
| `transients.yaml` | Hardware response | table, first-order, second-order, linear, rate-limited responses |
| `telemetry.yaml` | Output selection | aliases, source paths, units, plot definitions |

## Analysis YAML

```yaml
name: tca_mdot_controller
engine: engine.yaml
boundary_conditions: boundaries.yaml
operating_conditions: operating_conditions.yaml
controllers: controller.yaml
timings: timings.yaml
transients: transients.yaml
telemetry: telemetry.yaml
solver:
  transient:
    method: Radau
    rtol: 1.0e-7
    atol: 1.0e-6
    max_step: 0.005
analysis:
  type: tca_mdot_controller
  time: {start_s: 0.0, end_s: 8.0}
  output:
    csv: tca_mdot_controller.csv
    plot: tca_mdot_controller.png
```

Paths are resolved relative to the YAML file that owns the reference.

## Engine YAML

Engine YAML defines hardware that should remain stable across tests.

```yaml
name: two_valve_chain
components:
  methane_valve:
    type: Valve
    transient: methane_valve
    parameters:
      max_area: 2.5e-4
      discharge_coeff: 0.82
  methane_pipe:
    type: Pipe
    parameters:
      length: 0.4
      diameter: 0.03
      time_constant: 0.08
connections:
  - from: methane_valve.outlet
    to: methane_pipe.inlet
    domain: fluid
```

Maps are referenced by logical name through `analysis.yaml`:

```yaml
components:
  lox_pump:
    type: Pump
    parameters:
      pump_map:
        model: affinity_law
        mdot_design: 30.5
        dP_design: 13.0e6
        speed_design: 32000
    maps:
      head_map: {ref: lox_pump_affinity, output: pressure_rise}
      efficiency_map: {ref: lox_pump_affinity, output: efficiency}
```

## Maps

Use constants for simple values and CSV/HDF5 for test or simulation data.

```yaml
name: lox_pump_affinity
kind: structured_grid
source:
  type: csv
  path: lox_pump_affinity.csv
axes:
  - {name: corrected_speed, column: Nc}
  - {name: corrected_flow, column: Wc}
outputs:
  - {name: pressure_rise, column: dP}
  - {name: efficiency, column: eta}
interpolation:
  method: linear
  extrapolation: clamp
```

Multiple component map slots can use the same file by selecting different
outputs.

## Schedules And Profiles

Boundary conditions, operating conditions, timings, and table transients can use
schedules:

```yaml
schedule:
  type: table
  values:
    - [0.0, 40.0]
    - [10.0, 30.0]
    - [20.0, 40.0]
```

External target profiles can be CSV or JSON:

```yaml
targets:
  pms:
    schedule:
      type: profile
      source: {type: json, path: data/pms_targets.json}
      time_column: time_s
      outputs:
        mdot_total: mdot_total
        OF: OF
```

## Controllers

Operating-condition targets go through controllers before they become component
commands.

```yaml
controllers:
  lox_mdot_p:
    type: proportional
    inputs:
      target: targets.mdot_total
      measurement: measurements.mdot_total
    output: lox_valve.command
    parameters:
      bias: 0.5
      gain: 0.02
      lower_limit: 0.05
      upper_limit: 1.0
```

`timings.yaml` bypasses the controller layer and commands states directly:

```yaml
events:
  - target: main_lox_valve.command
    value: 1.0
```

## Transients

Transients define how hardware responds to commands:

```yaml
transients:
  main_lox_valve:
    type: first_order
    input: main_lox_valve.command
    output: main_lox_valve.position
    initial: 0.0
    parameters:
      time_constant: 0.2
      lower_limit: 0.0
      upper_limit: 1.0
```

Supported types are `table`, `first_order`, `second_order`, `linear`, and
`rate_limited`.

## Telemetry

Telemetry channels select source paths and assign output aliases:

```yaml
sample_rate_hz: 100
channels:
  - {alias: TIME, source: time, units: s}
  - {alias: MDOT_TOTAL, source: mdot.total, units: kg/s}
  - {alias: PC, source: chamber.P, units: bar}
exports:
  csv: true
  plot: true
plots:
  - title: Target Tracking
    x: TIME
    y: [TARGET_MDOT_TOTAL, MDOT_TOTAL]
```

Telemetry validation runs before solve, so every `source` must be in the
assembled source catalog.
