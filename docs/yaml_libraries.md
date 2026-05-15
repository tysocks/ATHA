# YAML Libraries And Includes

ATHA YAML files can reuse shared fragments with `include` or `$include`.
Includes are resolved relative to the YAML file that declares them, expanded
before schema validation, and can be used from analysis, engine, map,
transient, timing, controller, boundary-condition, operating-condition, and
telemetry files.

```yaml
include: ../library/common_valves.yaml
name: engine_with_local_overrides
components:
  main_lox_valve:
    parameters:
      CdA: 1.9e-4
```

Multiple includes are applied in order:

```yaml
$include:
  - ../library/base_engine.yaml
  - ../library/gg_branch.yaml
name: gas_generator_engine
```

Merge rules:

- mappings are deep-merged;
- local values override included scalar values;
- lists are appended, which is useful for `connections`, `channels`, and
  `events`;
- include cycles raise `ConfigError`.

Recommended library layout:

```text
configs/
  analysis.yaml
  engine.yaml
  telemetry.yaml
  library/
    valves.yaml
    pump_maps.yaml
    telemetry_common.yaml
```

Component template pattern:

```yaml
# library/valves.yaml
components:
  main_lox_valve:
    type: Valve
    parameters: {CdA: 1.9e-4}
  main_methane_valve:
    type: Valve
    parameters: {CdA: 1.2e-4}
```

Telemetry group pattern:

```yaml
# library/pump_telemetry.yaml
channels:
  - {alias: LOX_PUMP_DP, source: lox_pump.delta_P, units: MPa}
  - {alias: LOX_PUMP_PHI, source: lox_pump.phi}
  - {alias: LOX_PUMP_PSI, source: lox_pump.psi}
```

Then:

```yaml
$include: library/pump_telemetry.yaml
name: run_telemetry
sample_rate_hz: 100
channels:
  - {alias: TIME, source: time, units: s}
```

