# General Config Runner Architecture

Date: 2026-05-08

## Goal

ATHA should be able to run an analysis from a config folder without requiring a
custom Python runner for each example. The desired user-facing API is:

```python
from atha.runner import run_config_folder

result = run_config_folder("examples/17_tca_propellant_valve_transient/configs")
```

The same entrypoint should load:

- `analysis.yaml`
- `engine.yaml`
- map files
- transient files
- `boundaries.yaml`
- `operating_conditions.yaml`
- `controllers.yaml`
- `timings.yaml`
- `telemetry.yaml`

and then execute the requested analysis, solve the system, and export telemetry.

The long-term target is that changing YAML files changes the model behavior.
Python should only define reusable modelling capabilities, not example-specific
glue logic.

## Current Problem

The current `run.py` files are intentionally small, but the complexity has moved
into example-specific modules:

- `atha.examples.tca`
- `atha.examples.valve_volume`
- `atha.examples.two_valve_chain`
- `atha.examples.tca_valve_transient`

These modules repeat responsibilities:

- loading the analysis config;
- constructing a model dictionary from `engine.yaml`;
- evaluating boundaries and timings;
- integrating transient blocks;
- defining problem-specific ODE state order;
- calculating pressure, flow, and thrust;
- generating telemetry samples;
- choosing telemetry times;
- plotting.

This is better than large example `run.py` files, but it is still not the
architecture we want. It means adding a new model topology still requires a new
Python file.

## Desired Ownership

The YAML file boundaries should be:

### `engine.yaml`

Defines the physical model:

- component list;
- component type;
- component parameters, for example CdA, area, volume, pipe length, inertance,
  heat capacity, nominal pressure drop;
- component map bindings;
- component transient binding names;
- port layout and connections.

### `transients.yaml`

Defines actual component response to commands:

- `table`;
- `first_order`;
- `second_order`;
- `linear`;
- `rate_limited`;
- limits and initial state where appropriate;
- input command path and output state path.

### `timings.yaml`

Defines commanded event scripts:

- valve commands;
- ignition events;
- mode changes;
- discrete command targets;
- command schedules.

### `operating_conditions.yaml`

Defines operating targets that must be converted into commands by controllers:

- chamber pressure;
- thrust;
- total mass flow;
- mixture ratio;
- shaft speed;
- target temperatures or pressures.

### `controllers.yaml`

Defines how operating targets become commands:

- null/pass-through;
- splitters such as total mass flow plus OF to fuel/oxidizer flow;
- PID loops;
- gain/product gates;
- command selectors;
- saturation and rate limits where these are controller behavior rather than
  component hardware behavior.

### `boundaries.yaml`

Defines imposed external conditions:

- tank/supply pressures;
- inlet temperatures, densities, or enthalpies;
- ambient pressure;
- wall or environmental thermal conditions;
- fixed shaft speed only when the external test stand imposes it.

### `analysis.yaml`

Defines the run itself:

- analysis type;
- solver options;
- start/end time;
- initial solver state override;
- output filenames;
- output formats;
- trim/transient phase settings.

It should not contain physical component parameters except explicit analysis
overrides for sensitivity studies, sweeps, or Monte Carlo perturbations.

## Proposed User API

### Primary Function

```python
from atha.runner import run_config_folder

result = run_config_folder("path/to/configs")
```

Resolution rule:

- If the argument is a directory, load `analysis.yaml` inside that directory.
- If the argument is a file, load that file as the analysis manifest.

### Optional CLI

```powershell
atha-run examples/17_tca_propellant_valve_transient/configs
```

The CLI should call the same API and should not contain simulation logic.

## Proposed Internal Architecture

A single function is a good user-facing API, but not a good internal design.
The implementation should use a small number of generalized classes.

```text
run_config_folder()
  -> AnalysisRunner
      -> ConfigProcessor
      -> ComponentRegistry / EngineAssembler
      -> ScheduleProcessor
      -> ControllerProcessor
      -> TransientManager
      -> NetworkProblem
      -> SolverDriver
      -> OutputProcessor
```

### 1. `ConfigProcessor`

Responsibilities:

- load the analysis manifest;
- resolve all relative paths;
- validate required YAML files for the selected analysis type;
- validate references between engine components, maps, transients, controllers,
  timings, and telemetry;
- produce one immutable `LoadedAnalysisConfig`.

Current status:

- `atha.config.loader` already does part of this.
- It needs stronger validation and a folder-based loader convenience.

Requirements:

- `load_config_folder(path)`;
- strict schema validation for all known fields;
- clear errors for unknown component type, missing transient, missing telemetry
  source, and incompatible controller input;
- support YAML libraries, not only one-file-per-object references.

### 2. `ComponentRegistry`

Responsibilities:

- map component type strings to Python component classes;
- expose declared ports, states, algebraic variables, parameters, outputs, and
  accepted maps;
- prevent example code from hand-reading component parameters.

Current blocker:

- `atha.components.factory` is a narrow factory and does not understand all
  component types or map/transient bindings.
- Some example-only component types such as `GasVolume` and `OutletInertia` are
  represented as YAML concepts but not reusable components.

Requirements:

- central registry:

  ```python
  registry.register("Valve", ValveComponentSpec(...))
  ```

- component specs declare:
  - required parameters;
  - optional parameters;
  - ports;
  - state names;
  - output names;
  - transient-capable paths;
  - map-capable parameters;
  - units and scales.

### 3. `EngineAssembler`

Responsibilities:

- instantiate components from `engine.yaml`;
- attach maps;
- attach transient definitions to component state paths;
- compile the connection graph;
- build initial state and algebraic vectors.

Current status:

- `Engine.compile()` exists.
- Connection residuals exist, but they are not yet a complete global DAE solve.

Requirements:

- build all components through the registry;
- support connection validation before solve;
- support initial state values from both `engine.yaml` defaults and
  `analysis.yaml` solver overrides;
- expose a model-independent state/output path catalog.

### 4. `ScheduleProcessor`

Responsibilities:

- evaluate boundaries, timings, and operating target schedules at time `t`;
- support `constant`, `step`, `ramp`, `table`, `profile`, and runbox schedules;
- load CSV/JSON target profiles relative to the owning YAML file.

Current status:

- `atha.config.schedules` already implements much of this.

Requirements:

- unified call:

  ```python
  schedules.evaluate(t) -> ScheduleValues
  ```

- distinguish:
  - imposed boundary values;
  - command script values;
  - operating targets.

### 5. `ControllerProcessor`

Responsibilities:

- convert operating targets and timing commands into component command values;
- apply PID/null/splitter/gating logic;
- keep controller state where needed.

Current status:

- null, split, and gain/product controller helpers exist.
- There is no stateful PID controller integration.

Requirements:

- stateless and stateful controller interfaces;
- controller states registered into the global transient state vector where
  required;
- support controller output limits;
- validation that controller outputs target legal command paths.

### 6. `TransientManager`

Responsibilities:

- own transient blocks from `transients.yaml`;
- register transient states into the global state vector;
- evaluate actual component states from command values;
- compute derivatives for first-order, second-order, linear, rate-limited, and
  table responses.

Current status:

- `atha.config.transients.TransientSystem` exists and supports the requested
  scalar response types.
- It is not yet integrated into the main `EngineLayout` state registry.

Requirements:

- transient states must be first-class state variables;
- output paths such as `lox_valve.position` should be available to components
  and telemetry;
- table transients should be treated as prescribed values with no solver state;
- second-order transients require two states: value and rate;
- rate-limited and linear transients must handle command reversals cleanly.

### 7. `NetworkProblem`

Responsibilities:

- assemble the actual equations for the configured engine;
- evaluate component outputs and residuals;
- manage port pressure, enthalpy, temperature, density, and mass-flow unknowns;
- couple component maps, transients, boundaries, controllers, and operating
  targets.

This is the real heart of the architecture.

Current blocker:

- ATHA does not yet have a complete global port-variable DAE solve.
- The current compatibility solver can evaluate components and connection
  residuals, but many examples still hand-assemble simplified ODEs.
- Without a square system of algebraic port variables and residuals, arbitrary
  topologies cannot be solved only from `engine.yaml`.

Requirements:

- global unknown vector `Z` for algebraic port variables;
- global residual vector `Rz`;
- state derivative vector `dXdt`;
- consistent indexing and scaling;
- support for connections:
  - pressure equality or pressure drop equations;
  - mass conservation at junctions;
  - enthalpy/temperature propagation;
  - shaft torque balance;
  - thermal heat-flow balance;
- robust initialization and steady trim before transient integration;
- diagnostics naming each failed residual.

### 8. `SolverDriver`

Responsibilities:

- select and run steady-state, transient, sweep, or Monte Carlo analysis;
- call `NetworkProblem.evaluate`;
- run trim phases before transient phases when requested;
- manage solver tolerances and event handling.

Requirements:

- analysis type registry:

  ```yaml
  analysis:
    type: transient
  ```

- supported modes:
  - `steady`;
  - `transient`;
  - `profile`;
  - `sweep`;
  - `monte_carlo`;
- reusable initial-condition solve;
- explicit handling for discontinuities in timing schedules.

### 9. `OutputProcessor`

Responsibilities:

- validate telemetry sources before running;
- sample solver output at telemetry rate;
- export CSV/HDF5/plots;
- keep plotting generic.

Current problem:

- each example has its own `_plot` and `_telemetry_times` function.

Requirements:

- one telemetry sampler;
- one CSV writer;
- HDF5 writer with metadata and units;
- generic plot generation from telemetry groups, for example:

  ```yaml
  plots:
    - name: valve_positions
      x: TIME
      y: [METHANE_VALVE_POSITION, LOX_VALVE_POSITION]
    - name: chamber
      x: TIME
      y: [PC, THRUST]
  ```

Plotting should be YAML-driven and optional.

## Why One Function Alone Is Not Sufficient

A single public function is sufficient as an interface:

```python
run_config_folder(path)
```

But one internal function is not sufficient because the problem has distinct
responsibilities with different lifetimes:

- configs are loaded once;
- schedules are evaluated every time step;
- controllers may have state;
- transients have solver states;
- components have states, algebraic variables, and outputs;
- telemetry is sampled after solve;
- sweeps and Monte Carlo repeat the whole run many times.

Trying to implement all of this in one function would recreate the current
problem in a larger file. The right shape is one public function backed by a
small set of cohesive processors.

## What Is Impossible Today

### Arbitrary Engine Topology From YAML

Not fully possible yet.

Blocker:

- no completed global DAE/network solver for port pressure, mass flow, enthalpy,
  and residual equations;
- several components are still compute-output blocks rather than full residual
  equation providers;
- component factory/registry does not support all components uniformly;
- branch/junction behavior is not generalized.

Recommendation:

- implement arbitrary YAML execution in stages:
  1. serial feed systems with fixed boundary pressures;
  2. pressure-fed TCA networks;
  3. single-shaft gas-generator cycles;
  4. staged combustion cycles;
  5. full DAE networks with all coupled ports.

### Arbitrary Transient Physics From YAML

Partially possible.

Supported now:

- scalar transient response blocks.

Missing:

- transient states are not first-class `EngineLayout` states;
- component internals cannot yet declare complex transient submodels entirely
  from YAML;
- thermal/rotor/fluid storage transients still need component-specific
  equations.

Recommendation:

- integrate transient blocks into the global state registry before adding more
  transient types.

### Arbitrary Plots From Telemetry

Not fully possible.

Blocker:

- plotting is still example-owned.

Recommendation:

- add a generic plot section to `telemetry.yaml`.

## Implementation Plan

### Phase 1: Runner Skeleton

Deliverables:

- `atha.runner.run_config_folder(path)`;
- `ConfigFolderRunner`;
- resolve directory to `analysis.yaml`;
- reuse existing `load_analysis_config`;
- produce a standard `RunResult`;
- move `_telemetry_times` into a shared output sampler;
- add generic CSV export.

Acceptance criteria:

- examples can use:

  ```python
  from atha.runner import run_config_folder
  run_config_folder(Path(__file__).parent / "configs")
  ```

- no example-specific telemetry-time code remains.

### Phase 2: Generic Output Processor

Deliverables:

- telemetry validation before solve;
- generic sampling;
- generic CSV;
- generic plot definitions in telemetry YAML.

Acceptance criteria:

- examples 15, 16, and 17 no longer define `_plot`;
- adding a telemetry channel does not require Python edits.

### Phase 3: Component Registry

Deliverables:

- component spec registry;
- parameter validation;
- map binding validation;
- transient binding validation;
- replace ad hoc `_model_from_engine` functions.

Acceptance criteria:

- physical parameters are never manually extracted in example runners;
- unsupported component type errors include known valid types.

### Phase 4: Integrate Transients Into Engine State

Deliverables:

- transient block states appear in `layout.all_state_names()`;
- transient outputs are available to components as inputs;
- table transients are evaluated as prescribed commands;
- telemetry can read transient state paths without custom sample logic.

Acceptance criteria:

- valve position transients in examples 15, 16, and 17 are solved by the generic
  engine state path.

### Phase 5: Pressure-Fed TCA Network Solver

Deliverables:

- generic pressure-fed network analysis mode;
- valves compute flow from pressure drop and position;
- pipes provide pressure drop or inertance equations;
- injectors compute pressure drop and chamber inlet flow;
- chamber stores mass/pressure;
- nozzle computes outflow and thrust.

Acceptance criteria:

- examples 16 and 17 run from generic solver logic;
- no example-specific ODE functions remain for valve/chamber/nozzle chains.

### Phase 6: Full DAE Port Solve

Deliverables:

- global `Z` vector for port variables;
- connection and component residual assembly;
- Newton solve of algebraic network inside transient RHS;
- steady trim uses the same residual assembly;
- scaled residual diagnostics.

Acceptance criteria:

- gas-generator and staged-combustion examples can be represented as connected
  component networks instead of analysis-specific builder functions.

### Phase 7: Analysis Registry

Deliverables:

- `steady`;
- `transient`;
- `profile`;
- `sweep`;
- `monte_carlo`;
- `linearization`, later.

Acceptance criteria:

- analysis selection is YAML-driven;
- the public runner API does not change between analysis modes.

## Requirements List

### Functional Requirements

- Load a config folder from a single public function.
- Resolve all file references relative to owning YAML files.
- Run at least steady and transient analysis modes.
- Evaluate boundary, timing, operating, controller, and transient values at each
  solver step.
- Support map-backed component parameters.
- Export telemetry channels by alias and units.
- Support CSV output.
- Support HDF5 output with metadata.
- Support YAML-defined plots.
- Support initial condition solve/trim.
- Support Monte Carlo and sweep perturbations without editing engine YAML.

### Numerical Requirements

- Use scaled residuals.
- Provide named residual diagnostics.
- Support stiff integration with Radau.
- Warm-start algebraic solves.
- Detect discontinuities from timing schedules and handle them with phase
  breaks or solver event boundaries.
- Avoid duplicate `t=0` samples.
- Separate raw solver output from telemetry-resampled output.

### Schema Requirements

- Strong validation for unknown keys.
- Unit metadata on parameters and telemetry.
- Required/optional parameter declarations per component type.
- Component output catalog for telemetry validation.
- Clear path grammar:
  - `component.port.variable`;
  - `component.state`;
  - `component.output`;
  - `commands.*`;
  - `targets.*`;
  - `timings.*`;
  - `boundaries.*`.

### Testing Requirements

- Unit tests for config loading and schema validation.
- Unit tests for each transient type.
- Unit tests for generic telemetry sampling/export.
- Integration test for a pressure-fed TCA from YAML only.
- Integration test for valve timing discontinuities.
- Regression test for example 17 at approximately 150 kN steady thrust.
- Failure tests for missing references and invalid telemetry channels.

## Recommended Near-Term Direction

The next useful implementation should not add another example-specific runner.
It should create:

```text
atha/runner/
  __init__.py
  config_runner.py
  result.py

atha/output/
  sampling.py
  plotting.py
```

Then refactor examples 15, 16, and 17 first. These are small enough to prove the
runner concept, but rich enough to exercise timings, transients, boundaries,
engine parameters, telemetry, and plots.

Once those examples run through the generic path, move on to the TCA PMS example
and then the GG/FFSC examples.
