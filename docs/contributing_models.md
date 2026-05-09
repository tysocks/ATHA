# Adding Models And Outputs

This document describes the current extension workflow for ATHA's YAML-first
runner. The interfaces are intentionally conservative until the full arbitrary
port DAE solver is implemented.

## Add A Component Type

1. Create or update a component class under `atha/components/` if runtime object
   behavior is needed.
2. Register the component type in `atha/components/registry.py` with:
   - accepted parameters;
   - ports;
   - output paths;
   - state names;
   - map slots;
   - transient support flag;
   - optional residual contract.
3. Add telemetry source paths in `_component_source_paths` if the component has
   analysis-specific outputs.
4. Add config/schema tests for validation behavior.

Example registry shape:

```python
ComponentSpec(
    "Valve",
    optional_parameters=frozenset({"max_area", "CdA", "discharge_coeff"}),
    transient_capable=True,
    ports={"inlet": "fluid_in", "outlet": "fluid_out"},
    output_paths=("mdot", "position", "command"),
    residual_contract=residual_contract_for_type("Valve"),
)
```

## Add A Residual Provider

Residual providers live in `atha/components/residuals.py` and implement:

```python
class MyContract:
    def variables(self, component, model): ...
    def residuals(self, component, model): ...
    def evaluate(self, component, context): ...
```

Then register it through `residual_contract_for_type`. Residuals should be
named, scaled, and component-owned so `NetworkProblem` can report useful
diagnostics.

Current limitation: these contracts are component-local. Full connection-port
unknown generation is Phase 20.

## Add A Map

1. Add a logical map reference to `analysis.yaml`.
2. Add the map YAML file and any CSV/HDF5 data next to it.
3. Bind the component map slot in `engine.yaml`.

```yaml
maps:
  pump_combo: maps/pump_combo.yaml
```

```yaml
components:
  pump:
    type: Pump
    maps:
      head_map: {ref: pump_combo, output: pressure_rise}
      efficiency_map: {ref: pump_combo, output: efficiency}
```

## Add A Transient

Add a block in `transients.yaml` and reference it from `engine.yaml` when the
component owns that response.

```yaml
transients:
  lox_valve:
    type: rate_limited
    input: lox_valve.command
    output: lox_valve.position
    initial: 0.2
    parameters:
      opening_rate: 0.5
      closing_rate: 1.0
```

## Add A Controller

Prefer built-in controller types before adding Python functions:

- `null`
- `proportional`
- `pi`
- `pid`
- `scheduled_gain`
- `limiter`
- `rate_limiter`
- `selector`, `min`, `max`
- `of_mass_flow_split`

Python-function controllers are supported for examples, but built-in
controllers are easier to validate and serialize.

## Add An Analysis Type

Most users should not need to add Python runner code. If a case cannot be
represented by an existing `analysis.type`, add a small analysis handler and
register it in `atha/runner/analysis_registry.py`.

```python
registry.register(
    AnalysisSpec(
        type_name="my_analysis",
        handler=_run_my_analysis,
        mode="transient",
        description="Short user-facing description.",
    )
)
```

Handlers currently may accept the legacy `(config_path, output_dir)` signature.
New runner-facing code should prefer `AnalysisContext`, which carries the
loaded config, execution plan, analysis mode, output directory, and source
catalog helper in one object. Unsupported analysis types fail before solve and
report the known registered types plus close-match suggestions.

## Add A Telemetry Channel

1. Confirm the source exists in the source catalog.
2. Add the channel to `telemetry.yaml`.
3. Add it to a plot only if it is useful for review.

```yaml
channels:
  - alias: LOX_SHAFT_RPM
    source: lox_shaft.rpm
    units: rpm
```

If validation fails with `Telemetry source(s) are not available`, add the source
to the relevant component spec/catalog or correct the source path.
