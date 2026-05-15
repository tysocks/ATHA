# Contributor Guide

This guide covers the extension points most likely to matter when adding
components, map-backed physics, and validation cases.

## Residual Providers

Component residual providers live in `atha/components/residuals.py` and follow
the `ComponentResidualContract` protocol:

- `variables(component, model)` declares algebraic unknowns;
- `residuals(component, model)` declares residual equations and scales;
- `evaluate(component, context)` returns residual values and optional derived
  outputs.

Register providers in `atha/components/registry.py` through `ComponentSpec`.
Keep residual names stable because they become diagnostics and HDF5 paths.

Checklist:

- declare units and scales for every variable/residual;
- use `ResidualEvaluationContext.value()` for safe lookup;
- expose non-residual diagnostics as plain output paths;
- add residual closure coverage in `tests/unit/test_config.py` or a focused
  component test.

## Map-Backed Components

Map YAML files are built into `PerformanceMap` objects by
`atha.config.maps.build_performance_maps()`. Component map bindings use:

```yaml
maps:
  head_map: {ref: lox_pump_affinity, output: psi}
```

For pump head maps, prefer dimensionless inputs and outputs:

- `phi = mdot / (rho * omega * D^3)`;
- `psi = delta_P / (rho * omega^2 * D^2)`;
- `eta` or `efficiency`.

For passive valves, bind `cda_map` and return `CdA` from the map. The residual
provider supplies aliases such as `inlet.P`, `outlet.P`, `inlet.rho`, and
`position`.

## Controller Contributions

PID controllers should expose telemetry for:

- `controller.<name>.error`;
- `controller.<name>.integral`;
- `controller.<name>.derivative`;
- `controller.<name>.proportional_term`;
- `controller.<name>.integral_term`;
- `controller.<name>.derivative_term`;
- `controller.<name>.raw_command`;
- actuator command path.

When anti-windup is enabled, integral state is intentionally held during
saturation.

## Validation

Use the narrowest test that proves the new behavior:

- residual math: focused unit test;
- YAML/schema behavior: `tests/unit/test_config.py`;
- example behavior: integration regression window;
- output artifact behavior: CSV/HDF5 comparison utilities.

Before relying on a new example, run:

```powershell
python -m pytest tests\unit -q
python -m pytest tests\integration\test_example_regression.py -q
```

On Windows, if pytest cannot access the default temp root, set `TMP` and `TEMP`
to a workspace directory before running integration tests.

