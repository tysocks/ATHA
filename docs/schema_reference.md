# ATHA YAML Schema Reference

This is the concise public schema guide for config folders loaded by
`atha.runner.run_config_folder()`.

## `analysis.yaml`

Required:

- `name`: run name.
- `engine`: path to engine YAML.
- `analysis.type`: registered analysis type.

Common optional references:

- `maps`: name-to-path mapping.
- `boundary_conditions`: path.
- `operating_conditions`: path.
- `controllers`: path.
- `transients`: path or name-to-path mapping.
- `timings`: path.
- `telemetry`: path.

Common analysis fields:

```yaml
analysis:
  type: gg_single_shaft_transient
  time:
    start_s: 0.0
    end_s: 30.0
    phases:
      - {name: startup, start_s: 0.0, end_s: 3.0}
      - {name: CLC, start_s: 3.0, end_s: 25.0}
      - {name: shutdown, start_s: 25.0, end_s: 30.0}
  output:
    csv: run.csv
    plot: run.png
```

## `engine.yaml`

```yaml
name: engine_name
components:
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
connections:
  - {from: lox_pump.outlet, to: lox_pipe.inlet, domain: fluid}
```

Supported connection domains: `fluid`, `shaft`, `thermal`.

## `maps/*.yaml`

CSV-backed `phi`/`psi` pump map:

```yaml
name: lox_pump_affinity
kind: structured_grid
source: {type: csv, path: lox_pump_affinity.csv}
axes:
  - {name: phi, column: phi}
outputs:
  - {name: psi, column: psi}
  - {name: eta, column: eta}
interpolation: {method: linear}
```

## `controllers.yaml`

```yaml
name: controllers
evaluation: {frequency_hz: 10.0}
controllers:
  lox_generator_mdot_pid:
    type: pid
    active_phases: [CLC]
    inputs:
      target: targets.mdot_total
      measurement: measurements.mdot_total
    output: lox_generator_valve.command
    parameters:
      bias: 0.2
      proportional_gain: 0.26
      integral_gain: 0.3
      derivative_gain: 0.2
      lower_limit: 0.05
      upper_limit: 1.0
```

PID diagnostics include `error`, `integral`, `derivative`, `raw_command`,
`proportional_term`, `integral_term`, and `derivative_term`.

## `transients.yaml`

```yaml
name: transients
transients:
  main_lox_valve:
    type: first_order
    input: main_lox_valve.command
    output: main_lox_valve.position
    initial: 0.9
    parameters:
      time_constant: 0.15
      lower_limit: 0.0
      upper_limit: 1.0
```

Supported scalar transient types: `table`, `first_order`, `second_order`,
`linear`, `rate_limited`.

## `telemetry.yaml`

```yaml
name: telemetry
sample_rate_hz: 100
channels:
  - {alias: TIME, source: time, units: s}
  - {alias: PC, source: chamber.P, units: bar}
exports:
  csv: true
  plot: true
plots:
  - title: Chamber Pressure
    x: TIME
    y: [PC]
    ylabel: Pressure [bar]
```

## `timings.yaml`

```yaml
name: timings
events:
  - target: main_lox_valve.command
    schedule: {type: step, time: 25.0, initial: 1.0, final: 0.0}
```

## Boundary And Operating Conditions

Boundary conditions provide physical source values:

```yaml
conditions:
  nozzle.ambient.P: {value: 101325.0}
```

Operating conditions provide target schedules:

```yaml
targets:
  mdot_total:
    schedule:
      type: table
      values:
        - [0.0, 40.0]
        - [30.0, 40.0]
```

