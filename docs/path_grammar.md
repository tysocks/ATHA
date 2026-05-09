# ATHA Path Grammar

ATHA YAML files communicate through named source paths. Paths are strings; the
runner validates telemetry and controller references against the source catalog
assembled from the loaded configuration.

## Common Path Forms

| Form | Meaning | Example |
| --- | --- | --- |
| `time` | simulation/sample time | `time` |
| `component.state` | component state or output | `chamber.P` |
| `component.port.variable` | port-scoped value | `lox_pump.inlet.mdot` |
| `target.name` / `targets.name` | operating target | `target.mdot_total`, `targets.OF` |
| `timings.path` | timing-event value | `timings.main_valve.command` |
| `boundaries.path` | boundary-condition value | `boundaries.nozzle.ambient.P` |
| `controller.name.signal` | controller diagnostics | `controller.of_p.error` |
| `residuals.name` | residual diagnostic source | `residuals.nozzle.mdot_residual` |
| `connection.a.port__b.port.variable` | connection residual value | `connection.pipe.outlet__valve.inlet.P` |

## Component Paths

Component source paths are declared by component type. Examples:

- valve: `valve.command`, `valve.position`, `valve.mdot`
- pipe: `pipe.mdot`, `pipe.mdot_steady`, `pipe.P`, `pipe.dP`
- pump: `pump.mdot`, `pump.inlet.mdot`, `pump.delta_P`, `pump.power`
- rotor: `shaft.omega`, `shaft.rpm`
- nozzle: `nozzle.mdot`, `nozzle.thrust`
- chamber/preburner: `chamber.P`, `chamber.T`, `chamber.OF`, `chamber.mdot`

The full port-variable DAE solver will expand this grammar with complete
per-port thermodynamic histories.

## Controller Signal Lookup

Controller inputs use these prefixes:

```yaml
inputs:
  target: targets.mdot_total
  measurement: measurements.mdot_total
```

Supported lookup namespaces:

- `targets.*` from `operating_conditions.yaml`
- `timings.*` from `timings.yaml`
- `measurements.*` from live solver measurements
- previously computed command paths from earlier controllers

## Boundary And Timing Names

Boundary and timing target names should usually match the affected physical
path:

```yaml
conditions:
  lox_tank.outlet.P: {value: 4.0e5, units: Pa}

events:
  - target: main_lox_valve.command
    value: 1.0
```

Use `boundaries.<path>` only when a controller or telemetry channel explicitly
needs the boundary value itself.

## Connection Residual Paths

The connection residual path format is:

```text
connection.<source_component>.<source_port>__<target_component>.<target_port>.<variable>
```

Examples:

```text
connection.lox_pump.outlet__lox_pipe.inlet.P
connection.lox_turbine.shaft__lox_shaft.turbine_in.tau
```

These paths are mostly diagnostic today. Phase 20 will make them central to the
arbitrary port DAE solve.
