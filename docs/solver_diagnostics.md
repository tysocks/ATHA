# Solver Diagnostics And Common Failures

ATHA writes CSV, HDF5, plots, manifests, residual diagnostics, linearization
artifacts, and acceptance reports depending on the analysis type.

## Output Artifacts

Typical outputs in `outputs/`:

- `*.csv`: telemetry channels by alias
- `*.h5`: telemetry, state, algebraic, residual, and boundary groups where
  available
- `*.png`: telemetry plots
- `*.manifest.json`: artifact manifest and metadata
- `*_residuals.csv` / `*_residuals.json`: final named residual diagnostics
- `*.linearization.json`: `A`, `B`, `C`, `D`, labels, operating point
- `*.acceptance.json`: validation checks and pass/fail categories

## Residual Failures

`NetworkSolveError` reports the largest normalized residual:

```text
network solve 'case' failed residual tolerance 1e-8;
largest residual nozzle.mdot_residual=2.4e-3
```

Common causes:

- poor initial guesses;
- impossible boundary conditions;
- missing or wrong units;
- a valve/injector pressure drop that cannot support the requested flow;
- telemetry or controller paths that refer to old component names;
- map extrapolation outside the intended range.

## Telemetry Source Errors

```text
ValueError: Telemetry source(s) are not available: [...]
```

Fixes:

- correct the `source` path in `telemetry.yaml`;
- add the output path to the component source catalog;
- verify the component exists in `engine.yaml`;
- verify the analysis runner actually samples that source.

## Controller Tracking Errors

If a controller appears inactive:

- check the controller input paths;
- check command saturation at `lower_limit` or `upper_limit`;
- compare `*.command` and `*.position` telemetry;
- check transient time constants;
- verify sign convention. A positive proportional gain assumes increasing the
  command increases the measured value.

Example 19 intentionally exports both command and position for the controlled
crossover valves to make this visible.

## Stiff Or Slow Transients

For stiff dynamics:

- use `Radau`;
- reduce `max_step` near scheduled events;
- add event breakpoints through table/step/ramp schedules;
- keep state magnitudes and tolerances physically meaningful.

Current compatibility runners do not yet consume every `ExecutionPlan` recovery
hook. Full recovery behavior belongs with the Phase 20 universal DAE loop.

## Acceptance Reports

Acceptance reports classify failures by category:

- `numerical`
- `physical_model`
- `controller`
- `telemetry`
- `linearization`

Example:

```json
{
  "name": "tail_mdot_rms_tracking",
  "category": "controller",
  "passed": true,
  "value": 0.0236,
  "limit": 0.28
}
```

This lets a failed run report whether the issue is numerical convergence,
target tracking, physical response, output plumbing, or missing linearization.
