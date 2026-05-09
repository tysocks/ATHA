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

- [x] `atha.runner.run_config_folder(path)`;
- [x] `ConfigFolderRunner`;
- [x] resolve directory to `analysis.yaml`;
- [x] reuse existing `load_analysis_config`;
- [x] produce a standard `RunResult`;
- [x] move `_telemetry_times` into a shared output sampler;
- [x] add generic CSV export through the existing telemetry writer.

Acceptance criteria:

- [x] examples can use:

  ```python
  from atha.runner import run_config_folder
  run_config_folder(Path(__file__).parent / "configs")
  ```

- [x] no example-specific telemetry-time code remains in examples 15, 16, 17,
  and 18.

Phase 1 implementation notes:

- Added `atha.runner.run_config_folder`, `ConfigFolderRunner`, and `RunResult`.
- Added shared `atha.output.sampling.telemetry_times`.
- Refactored examples 15-18 to call the public config runner.
- The runner still dispatches by analysis type to current implementations. This
  is intentional for Phase 1; generic network solving begins in later phases.

### Phase 2: Generic Output Processor

Deliverables:

- [x] telemetry validation before solve for examples 15-18 using source
  catalogs;
- [x] generic sampling;
- [x] generic CSV;
- [x] generic plot definitions in telemetry YAML.

Acceptance criteria:

- [x] examples 15, 16, 17, and 18 no longer define `_plot`;
- [x] adding a telemetry channel does not require Python edits when the source
  is already present in the runner source catalog.

Phase 2 implementation notes:

- Added `atha.output.plotting.plot_telemetry`.
- Added `validate_telemetry_sources`.
- Added `plots:` blocks to examples 15-18 telemetry YAML files.
- Existing limitation: source catalogs are still provided by the current
  analysis-specific implementations. A fully generic source catalog depends on
  Phase 3 component registry and Phase 5 network problem work.

### Phase 3: Component Registry

Deliverables:

- [x] component spec registry;
- [x] parameter validation for registered component specs;
- [x] map binding validation;
- [x] transient binding validation;
- [x] replace ad hoc `_model_from_engine` functions.

Acceptance criteria:

- [x] physical parameters are never manually extracted in example runners for
  examples 15, 16, 17, and 18;
- [x] unsupported component type errors include known valid types.

Phase 3 implementation notes:

- Added `atha.components.registry` with `ComponentSpec`, `ComponentRegistry`,
  `validate_engine_config`, `extract_engine_model`, and known-type reporting.
- `load_analysis_config` now validates engine component types and transient
  bindings through the registry after resolving referenced YAML files.
- Examples 15-18 now use `extract_engine_model(loaded.engine)` instead of local
  `_model_from_engine` helpers.
- Complex legacy components are registered with permissive parameter validation
  until their component specs are filled out in later phases.

Remaining limitation:

- The registry currently provides metadata and temporary model extraction. It
  does not yet instantiate all components or provide a complete output/source
  catalog. Those require Phase 4 and Phase 5 work.

### Phase 4: Integrate Transients Into Engine State

Deliverables:

- [x] transient block states appear in `layout.all_state_names()`;
- [x] transient outputs are available to components as inputs through
  `EngineLayout.evaluate(..., U)`;
- [x] table transients are evaluated as prescribed commands with no solver
  state;
- [x] telemetry can read transient state paths without custom sample logic.

Acceptance criteria:

- [x] valve position transients in examples 15, 16, and 17 are exposed through
  the generic transient state/source path.

Phase 4 implementation notes:

- Added `atha.components.transient.TransientBlockComponent`, an adapter that
  registers scalar transient blocks as normal `BaseComponent` instances.
- Added `TransientSystem.build_layout()` so transient blocks can compile into an
  `EngineLayout`. For example, a block with output
  `methane_valve.position` appears in `layout.all_state_names()` as
  `methane_valve.position`.
- Added transient `state_names()`, `source_catalog()`, and `sample_sources()`.
  These are now the telemetry-facing interface for transient outputs and
  internal state values.
- Refactored examples 15, 16, 17, and 18 to validate transient telemetry from
  `TransientSystem.source_catalog()` and sample transient paths through
  `TransientSystem.sample_sources()`.
- Added unit coverage proving first-order/linear, second-order, and table
  transients expose the expected state/source paths.

Remaining limitation:

- Examples 15-18 still use their compatibility ODE functions for pressure,
  flow, chamber, and nozzle physics. Phase 4 makes transient states first-class;
  Phase 5 is where those transient outputs are consumed by a generic
  pressure-fed network problem instead of analysis-specific RHS code.

### Phase 5: Pressure-Fed TCA Network Solver

Deliverables:

- [x] generic pressure-fed network analysis mode;
- [x] valves compute flow from pressure drop and position;
- [x] pipes provide first-order flow inertia/time-lag equations;
- [x] injectors contribute fixed nominal pressure drop before chamber inlet
  flow;
- [x] chamber stores mass/pressure;
- [x] nozzle computes outflow and thrust.

Acceptance criteria:

- [x] examples 16 and 17 run from generic solver logic;
- [x] no example-specific ODE functions remain for valve/chamber/nozzle chains.

Phase 5 implementation notes:

- Added `atha.analysis.pressure_fed.run_pressure_fed_tca`, a reusable
  pressure-fed chamber/nozzle transient solver.
- The solver discovers feed legs from `engine.yaml` as
  `Valve -> Pipe -> MassFlowInjector -> CombustionChamber` chains instead of
  hard-coding methane/LOX or valve A/B topology.
- The state vector is assembled as chamber pressure, one pipe mass-flow state
  per discovered leg, and the transient block state vector from
  `TransientSystem`.
- Valve flow is computed from supply pressure, chamber pressure, injector
  nominal pressure drop, density, and actual valve position. Pipe flow uses the
  YAML pipe `time_constant`. Chamber pressure uses compressible storage through
  `gas_R`, `gas_T`, and chamber volume. Nozzle mass flow and thrust use the
  nozzle conductance, throat area, and thrust coefficient from `engine.yaml`.
- `ConfigFolderRunner` now dispatches `two_valve_transient_chain`,
  `tca_propellant_valve_transient`, and `tca_mdot_controller` to the generic
  pressure-fed solver.
- `atha.examples.two_valve_chain` and `atha.examples.tca_valve_transient` are
  now compatibility wrappers around the generic solver rather than owning their
  own ODE systems.
- Added unit coverage for discovering pressure-fed feed legs from YAML
  connections.

Remaining limitation:

- This is a pressure-fed serial-leg solver, not the full DAE port-variable
  solver. It supports the currently maintained TCA valve/chamber/nozzle
  examples but does not yet solve arbitrary junctions, branch networks, pump
  maps, shaft coupling, thermal coupling, or algebraic pressure nodes. Those
  remain Phase 6 work.

### Phase 6: Full DAE Port Solve

Deliverables:

- [x] global `Z` vector for pressure-fed network algebraic variables;
- [x] connection and component residual assembly for pressure-fed leg/nozzle
  algebraics;
- [x] Newton solve of algebraic network inside transient RHS;
- [x] steady trim uses the same pressure-fed residual assembly;
- [x] scaled residual diagnostics.

Acceptance criteria:

- [ ] gas-generator and staged-combustion examples can be represented as
  connected component networks instead of analysis-specific builder functions.

Phase 6 implementation notes:

- Added `atha.solver.algebraic` with `AlgebraicNetworkProblem`,
  `AlgebraicVariable`, `AlgebraicResidual`, and `AlgebraicSolution`.
- The algebraic solver owns ordered variable names, residual names, scaling,
  root solving, fallback least-squares solving, and maximum normalized residual
  diagnostics.
- Updated `atha.analysis.pressure_fed` to build a named algebraic problem for
  pressure-fed TCA networks:
  - one `Z` variable for each discovered leg steady mass flow, for example
    `methane_pipe.mdot_steady`;
  - one `Z` variable for `nozzle.mdot`;
  - matching residuals for valve/injector leg flow and nozzle flow.
- The pressure-fed transient RHS now solves the algebraic network at each RHS
  call and uses the solved `Z` values for pipe flow targets and nozzle outflow.
- Added `solve_pressure_fed_steady_state`, which trims chamber pressure using
  the same algebraic residual assembly plus chamber mass balance.
- Added unit coverage for the named `Z` solve and steady trim behavior.

Remaining limitation:

- Phase 6 now exists for pressure-fed serial-leg TCA networks. It is not yet the
  full arbitrary DAE solver needed for gas-generator and staged-combustion
  cycles. Remaining work includes global port pressure/enthalpy unknowns,
  junction mass conservation, pump/turbine map residuals, shaft torque balance,
  thermal coupling residuals, robust warm-started algebraic solves across all
  component classes, and steady trim for those coupled networks.

### Phase 6 Acceptance Scope: FFSC DAE Case

Added acceptance example:

- [x] `examples/19_ffsc_dae_acceptance`;
- [x] FFSC methalox engine definition in YAML;
- [x] 150 kN design target in `analysis.yaml`;
- [x] operating targets in `operating_conditions.yaml`;
- [x] controller binding in `controller.yaml`;
- [x] controller implementation in local `controller.py`;
- [x] four valve transient definitions in `transients.yaml`;
- [x] fixed main valve commands in `timings.yaml`;
- [x] pump and turbine affinity-law map assets in `configs/maps`;
- [x] telemetry channel request in `telemetry.yaml`;
- [x] validation runner proving the YAML loads and target schedules evaluate.

Acceptance model topology:

- LOX side:
  `lox_pump -> pipe -> lox_splitter`.
- Methane side:
  `methane_pump -> pipe -> methane_splitter`.
- LOX splitter branches:
  - majority branch through `main_lox_valve` to `ox_preburner`;
  - crossover branch through `lox_crossover_valve` to `fuel_preburner`.
- Methane splitter branches:
  - majority branch through `main_methane_valve` to `fuel_preburner`;
  - crossover branch through `methane_crossover_valve` to `ox_preburner`.
- Ox-rich path:
  `ox_preburner -> pipe -> lox_turbine -> pipe -> main_lox_injector ->
  chamber`.
- Fuel-rich path:
  `fuel_preburner -> pipe -> methane_turbine -> pipe ->
  main_methane_injector -> chamber`.
- Shaft paths:
  `lox_turbine -> lox_shaft -> lox_pump` and
  `methane_turbine -> methane_shaft -> methane_pump`.

Target schedules:

- `mdot_total`: 40 kg/s from 0-10 s, 30 kg/s from 10-20 s, 40 kg/s
  from 20-25 s.
- `OF`: 3.0 through 5 s, ramping to 3.4 at 25 s.

Controllers:

- Total mass flow target drives `methane_crossover_valve.command` with a
  proportional controller, initial crossover valve position 50%.
- OF target drives `lox_crossover_valve.command` with a proportional
  controller, initial crossover valve position 50%.
- `main_lox_valve` and `main_methane_valve` are fixed at 100% command.

Remaining implementation actions for this acceptance case:

- [ ] Extend the analysis registry so `analysis.type: ffsc_dae_transient` is
  dispatched through the public runner.
- [ ] Build a global DAE unknown catalog from arbitrary `engine.yaml`
  connections:
  - port pressures;
  - port mass flows;
  - temperature or enthalpy;
  - shaft speed and torque variables where algebraic coupling is required.
- [ ] Add component residual providers for:
  - `Pump` map pressure rise, efficiency, torque load, and flow compatibility;
  - `FlowSplitter` mass conservation and branch split residuals;
  - `Pipe` inertance/friction residuals;
  - `Valve` flow residuals using transient valve position;
  - `MassFlowInjector` pressure-drop and flow residuals;
  - `Preburner` mass, mixture-ratio, pressure, and temperature residuals;
  - `Turbine` pressure ratio, efficiency, torque drive, and outlet state;
  - `Rotor` shaft torque balance and speed dynamics;
  - `CombustionChamber` mass/pressure/mixture residuals;
  - `Nozzle` throat mass flow and thrust residuals.
- [ ] Add junction handling for splitters and merged chamber/preburner inlet
  flows.
- [ ] Add warm-started algebraic solves at each transient RHS call using the
  same residual names and scales exposed in diagnostics.
- [ ] Add steady trim for the FFSC cycle using the same residual assembly before
  transient integration.
- [ ] Feed live measurements to the controllers:
  - total engine mass flow;
  - chamber OF;
  - valve positions;
  - shaft speeds.
- [ ] Export all requested telemetry paths from the generic output catalog.
- [ ] Acceptance pass criteria:
  - example 19 runs from `run_config_folder`;
  - `mdot_total` follows the 40 -> 30 -> 40 kg/s target profile;
  - OF ramps from 3.0 to 3.4;
  - controlled valve commands move in response to target error;
  - preburner flow changes alter turbine torque;
  - shaft speeds respond dynamically;
  - pump pressure rise changes with shaft speed through the affinity maps;
  - chamber/nozzle thrust settles near 150 kN during 40 kg/s operation.

### Phase 7: Analysis Registry

Deliverables:

- [ ] `steady`;
- [x] `transient`;
- [ ] `profile`;
- [ ] `sweep`;
- [ ] `monte_carlo`;
- [ ] `linearization`, later.

Acceptance criteria:

- [x] analysis selection is YAML-driven;
- [x] the public runner API does not change between analysis modes.

Phase 7 implementation notes:

- Added `atha.runner.analysis_registry` with:
  - `AnalysisSpec`;
  - `AnalysisRegistry`;
  - `DEFAULT_ANALYSIS_REGISTRY`.
- Replaced the hard-coded `ConfigFolderRunner._run_analysis` dispatch chain
  with registry lookup by `analysis.type`.
- Registered currently supported analysis types:
  - `valve_volume_transient`;
  - `two_valve_transient_chain`;
  - `tca_propellant_valve_transient`;
  - `tca_mdot_controller`;
  - `ffsc_dae_transient`.
- Added `analysis_mode` metadata to `RunResult`.
- Added `atha.analysis.ffsc_acceptance.validate_ffsc_dae_acceptance`, allowing
  example 19 to execute through `run_config_folder` as a configuration and
  schedule validation acceptance case while the full arbitrary FFSC DAE solve is
  still pending.
- Refactored `examples/19_ffsc_dae_acceptance/run.py` to use the public runner.
- Added unit coverage for registered analysis types and public-runner dispatch.

Remaining limitation:

- The registry now supports clean YAML-driven selection, but only transient
  handlers are implemented. `steady`, `profile`, `sweep`, `monte_carlo`, and
  `linearization` still need concrete registry handlers backed by shared solver
  drivers rather than legacy example scripts.

## 2026-05-09 ROCETS Alignment Audit

After reviewing the current codebase against this architecture document and the
NASA ROCETS final report in `resources/19910011919.pdf`, ATHA has implemented a
large part of the outer configuration framework but still lacks the numerical
center that made ROCETS broadly useful.

ROCETS separated responsibilities into a library system, Run Processor,
Execution Processor, and Output Processor. The Run Processor interpreted
high-level experiment inputs, schedules, user algebraic balances, and
integration options. The Execution Processor controlled looping, print
selection, balancing, transient integration, and linearization. The Output
Processor let users select print and plot parameters independently from the
model. ROCETS supported three main run modes: steady-state trim balance,
transient operation, and linearization. Its transient integration closed
corrector equations and algebraic balances together with a modified
Newton-Raphson iteration. It also supported state activation modes similar to
`ON`, `OFF`, and `STEADY-STATE`, which allowed dynamic states to be frozen or
forced to steady-state during a transient.

ATHA is now aligned with ROCETS at the high-level input organization:

- modular YAML files own engine, maps, transients, boundaries, operating
  targets, controllers, timings, telemetry, and analysis settings;
- `run_config_folder()` is the single public entrypoint;
- analysis selection is registry-backed;
- scalar transient blocks exist and are first-class state paths;
- pressure-fed TCA examples run from a reusable network solver;
- example 19 captures the target FFSC acceptance topology in YAML.

The remaining work is to replace compatibility solvers with a general execution
processor and global residual engine.

### Remaining Gaps

#### Run Processor Gaps

- No complete `load_config_folder()` API in `atha.config`; the runner resolves
  folders, but config processing itself still centers on an Analysis YAML file.
- Schema validation is permissive in several places. Unknown keys can still
  pass silently, especially in analysis settings, controllers, component
  parameters with `allow_extra_parameters`, and telemetry plot definitions.
- There is no YAML include/library mechanism for reusable subsystem fragments,
  controller libraries, or repeated component templates.
- User-defined algebraic balances are not represented in YAML. ROCETS allowed
  schedules to request targets such as chamber pressure and mixture ratio while
  balances varied variables such as valve areas.
- Integration options do not yet support defaults plus per-state exceptions.
- State activation modes are missing. ATHA cannot yet mark a state as active,
  inactive, fixed, or forced to steady-state during a transient.

#### Component Library Gaps

- Component registry entries validate only a small parameter subset. They do not
  yet declare full ports, units, states, algebraic variables, map slots,
  transient-capable paths, output names, or residual names.
- Many components are still `compute_outputs()` blocks rather than residual
  providers. That means insertion order still matters and algebraic loops are
  not solved generically.
- Pump and turbine maps exist in simplified form, but map-backed residual
  behavior is not generalized for corrected speed, corrected flow, efficiency,
  pressure ratio, torque, and power balance.
- `FlowSplitter`, junctions, merged inlets, preburners, turbines, shafts, and
  thermal components do not yet provide the residual contracts needed for
  arbitrary FFSC or GG topology solving.
- Units and scaling are not enforced at the component parameter boundary.

#### Execution Processor Gaps

- The current global algebraic solve is limited to pressure-fed TCA leg/nozzle
  algebraics. It is not a complete port-variable DAE solve.
- There is no generic `NetworkProblem` that assembles:
  - state vector `X`;
  - algebraic vector `Z`;
  - residual vector `Rz`;
  - command vector `U`;
  - output/source catalog;
  - residual scales and diagnostics.
- Connection residuals exist in `EngineLayout`, but arbitrary layouts are not
  solved by a warm-started algebraic loop inside transient integration.
- There is no robust steady trim for arbitrary networks before transient
  integration.
- Discontinuities from schedules and timings are not split into integration
  phases, so stiff solvers can step across command changes without controlled
  restart behavior.
- Linearization is not implemented. ROCETS treated linear partial generation as
  a first-class run mode.
- Controller state is not part of the global state vector. PID, integrator
  anti-windup, command limiting, and controller diagnostics are still missing.

#### Output Processor Gaps

- CSV and YAML-driven plots exist, but generic HDF5 output with metadata,
  residuals, commands, controller signals, units, and run provenance is missing.
- Telemetry validation depends on analysis-specific source catalogs.
- Residual diagnostics are not exported as first-class telemetry or failure
  reports.
- There is no standard comparison report against test data or reference
  simulation data.

#### Validation Gaps

- Example 19 validates as YAML, but does not yet execute as a coupled FFSC
  transient.
- Examples 04, 09, 10, and 13 still contain substantial example-specific Python
  implementation logic.
- There is no ROCETS-like TTBE acceptance model with documented trim, throttle,
  start, shutdown, and linearization validation cases.
- There is no regression suite that asserts residual closure, controller target
  tracking, shaft speed response, pump/turbine map response, and telemetry
  export quality for coupled staged-combustion cycles.

## Completion Roadmap

The remaining implementation should proceed in phases that align ATHA with the
ROCETS processor model while preserving the current YAML-first user interface.

### Phase 8: Run Processor Hardening

Deliverables:

- [x] Add `atha.config.load_config_folder(path)` as the config-layer equivalent
  of runner folder resolution.
- [x] Add strict schema validation for known YAML sections.
- [x] Add clear unknown-key errors for top-level YAML sections.
- [ ] Add reusable YAML library support for:
  - component templates;
  - map libraries;
  - controller libraries;
  - telemetry channel groups;
  - subsystem fragments.
- [ ] Add user-defined algebraic balance declarations, for example:

  ```yaml
  balances:
    chamber_pressure:
      residual: chamber.P - targets.Pc
      variable: lox_crossover_valve.command
      scale: 5.0e6
  ```

- [ ] Add integration option defaults plus per-state exceptions.
- [ ] Add state mode declarations:
  - `active`;
  - `inactive`;
  - `fixed`;
  - `steady_state`.
- [x] Validate controller outputs against legal command or transient input
  paths.

Acceptance criteria:

- [x] Invalid YAML keys fail before solve with actionable messages.
- [ ] Example 19 can declare algebraic balances for `mdot_total` and `OF`
  without Python changes.
- [ ] A config folder can reuse a valve template or telemetry channel group
  from another YAML file.

Phase 8 implementation notes:

- Added `atha.config.load_config_folder(path)`, which resolves directories to
  `analysis.yaml` at the config layer instead of only in the runner.
- Added unsupported-key checks for:
  - `analysis.yaml` top-level keys;
  - `engine.yaml` top-level keys;
  - component blocks;
  - map files;
  - transient files;
  - boundary-condition files;
  - operating-condition files;
  - timing files;
  - controller files;
  - telemetry files.
- Kept intentionally extensible sections flexible, including component
  `parameters`, `analysis.analysis`, map interpolation/scaling details, and
  connection metadata.
- Added controller shape validation for current controller types:
  - `null`;
  - `of_mass_flow_split`;
  - `gain_product`;
  - `python_function`.
- Added validation that controller outputs either target a known transient
  command path or use an explicit extension namespace such as `commands.*`.
- Moved example 19 acceptance notes from a top-level `acceptance:` YAML block
  into `analysis.acceptance` so strict top-level Analysis YAML validation stays
  meaningful.
- Added unit coverage for folder loading, unknown-key failures, and invalid
  controller output paths.

Remaining limitation:

- Phase 8 is only partially complete. Reusable YAML libraries, user-defined
  algebraic balances, integration defaults/exceptions, and state activation
  modes still need implementation.

Additional implementation notes:

- Built-in proportional controllers are now supported and validated. This
  removed the need for example 18 to load `controller.py` for the simple
  mass-flow control case.

Remaining blockers:

- YAML library support is still open because it needs a merge/override policy
  for templates, subsystem fragments, and telemetry groups. Implementing it
  without that policy would make config precedence ambiguous.
- User-defined algebraic balances are still open because the expression parser
  and safe variable binding need to be tied to the Phase 10/11 network residual
  system. String expressions should not be evaluated ad hoc.
- Per-state integration exceptions and state modes are still open because the
  current compatibility solvers do not yet expose a unified global state
  registry for all physical, transient, and controller states.

### Phase 9: Engine Assembler And Source Catalog

Deliverables:

- [ ] Replace temporary `extract_engine_model()` usage with a true
  `EngineAssembler`.
- [ ] Instantiate every component from `engine.yaml` through the component
  registry.
- [ ] Attach map objects to component map slots during assembly.
- [ ] Attach transient components to command/output paths during assembly.
- [x] Build a model-independent source catalog for:
  - states;
  - algebraics;
  - commands;
  - boundaries;
  - targets;
  - controller outputs;
  - transient outputs;
  - component outputs;
  - residuals.
- [x] Validate telemetry channels against this catalog before running where
  current runners expose enough model metadata.
- [x] Generate initial `X` and `Z` vectors from engine defaults and
  `analysis.yaml` overrides.

Acceptance criteria:

- [ ] Examples 15-18 no longer require analysis-specific source catalog
  functions.
- [ ] Example 19 produces a complete source catalog, including shaft speeds,
  crossover valve positions, pump pressure rise, turbine torque, residuals, and
  controller signals.
- [ ] Component connection errors identify the exact invalid port path.

Phase 9 implementation notes:

- Added `atha.assembly.EngineAssembler` and `SourceCatalog`.
- The assembler now builds a pre-solve flat source catalog from
  `LoadedAnalysisConfig`, including:
  - `time`;
  - boundary-condition paths;
  - operating target paths, including profile output aliases such as
    `target.mdot_total`;
  - timing command targets;
  - controller output paths;
  - transient command, output, and state paths;
  - component output/source heuristics for current component types;
  - connection residual/source paths.
- Added assembler-backed telemetry source validation in:
  - `atha.examples.valve_volume`;
  - `atha.analysis.pressure_fed`;
  - `atha.analysis.ffsc_acceptance`.
- Added unit coverage proving example 19 source paths are present, including
  crossover valve commands/positions, shaft RPM, chamber pressure, and nozzle
  thrust.

Remaining limitation:

- This is the source-catalog slice of Phase 9, not the full engine assembly
  replacement. Current compatibility solvers still use `extract_engine_model()`
  and some analysis-specific telemetry supplements. Full completion requires
  instantiating all components, attaching maps/transients, producing universal
  initial `X`/`Z`, and deriving component output/residual paths from registry
  specs rather than heuristics.

Additional implementation notes:

- Added `EngineAssembler.initial_vectors()`, which collects component
  `initial_state`, transient initial states, `analysis.initial_state`
  overrides, component-contract algebraic names, and
  `analysis.initial_algebraic` overrides into generic `X/Z` vectors.

Remaining blockers:

- Full replacement of `extract_engine_model()` remains blocked by missing
  component constructors/map attachment for the complete component library.
  The assembler can now produce metadata and initial vectors, but several
  components still require compatibility model dictionaries.

### Phase 10: Generic Global DAE Network Problem

Deliverables:

- [x] Add `atha.network.NetworkProblem`.
- [ ] Build a global algebraic unknown vector `Z` from connected port variables:
  - pressure;
  - mass flow;
  - enthalpy or temperature;
  - density where required;
  - shaft torque/speed algebraics;
  - thermal heat-flow algebraics.
- [x] Build a named/scaled global residual vector `Rz` for registered residual
  equations.
- [x] Support warm-started algebraic solves at each transient RHS call.
- [x] Detect non-square systems and report missing or over-specified residuals.
- [ ] Detect algebraic loops explicitly and solve them rather than relying on
  component insertion order.
- [x] Add residual scaling rules and named largest-residual diagnostics.
- [x] Support sparse Jacobian hints.

Acceptance criteria:

- [x] A generic pressure-fed TCA runs through `NetworkProblem` instead of
  `atha.analysis.pressure_fed` compatibility equations.
- [x] Example 16 and 17 outputs remain within current regression tolerances.
- [x] Failure diagnostics name the largest unclosed residual and associated
  component/connection path.

Phase 10 implementation notes:

- Added `atha.network.NetworkProblem`, `NetworkVariable`,
  `NetworkResidual`, `NetworkSolution`, `WarmStart`, and
  `NetworkStructureError`.
- `NetworkProblem` owns algebraic vector labels, residual labels, scaling,
  root/least-squares fallback, optional sparse Jacobian hints, and largest
  normalized residual diagnostics.
- The network layer now rejects non-square systems and catches residual
  evaluator mismatches where declared residuals are missing or unexpected
  residuals are returned.
- `atha.solver.algebraic` now re-exports the new network classes for backwards
  compatibility.
- The pressure-fed TCA algebraic solve now builds a `NetworkProblem` directly.
  This means examples 16 and 17 exercise the new Phase 10 algebraic layer while
  retaining their current pressure/chamber transient equations.
- Added unit coverage for warm-started solves, non-square detection, and
  missing/extra residual diagnostics.
- Re-ran examples 16 and 17 after the `NetworkProblem` migration; outputs
  remain consistent with the current regression behavior.

Remaining limitation:

- Phase 10 is structurally in place, but the full ROCETS-like DAE network is
  not complete until Phase 11 component residual contracts exist. Today the
  pressure-fed TCA creates a named `Z/Rz` system manually. Arbitrary connected
  port variables, loop discovery, shaft/thermal algebraics, and automatic
  component residual assembly still depend on the component residual-provider
  work in Phase 11.

### Phase 11: Component Residual Contracts

Deliverables:

- [x] Define a residual-provider interface for all reusable components.
- [ ] Implement residual contracts for:
  - [x] `Valve`;
  - [ ] `Pipe`;
  - [x] `MassFlowInjector`;
  - [x] `FlowSplitter`;
  - [ ] `CombustionChamber`;
  - [ ] `Preburner`;
  - [x] `Nozzle`;
  - [x] `Pump`;
  - [x] `Turbine`;
  - [x] `Rotor`;
  - `RegenChannel`;
  - thermal nodes.
- [x] Extend component registry specs with:
  - [x] required and optional parameters;
  - [ ] units;
  - [x] ports and domains;
  - [x] state names;
  - [x] algebraic variables;
  - [x] residual names;
  - [x] output paths;
  - [x] accepted map slots;
  - [x] transient-capable inputs.
- [ ] Make pump maps support corrected speed, corrected flow, head or pressure
  rise, efficiency, and optional extrapolation rules.
- [ ] Make turbine maps support pressure ratio, corrected flow, efficiency, and
  torque/power residuals.
- [ ] Add branch and merge residuals for preburner/chamber multi-inlet mixing.

Acceptance criteria:

- [x] Example 19 component graph assembles into a square residual system for
  registered component-local residual contracts.
- [ ] Pump and turbine map values change with shaft speed and flow.
- [x] Shaft acceleration responds to turbine drive torque minus pump load torque
  in the existing `Rotor` component model.

Phase 11 implementation notes:

- Added `atha.components.residuals`, including:
  - `ComponentResidualContract`;
  - `ResidualEvaluationContext`;
  - `ValveFlowContract`;
  - `InjectorPressureDropContract`;
  - `FlowSplitterContract`;
  - `NozzleConductanceContract`;
  - `PumpHeadContract`;
  - `TurbinePowerContract`;
  - `RotorTorqueBalanceContract`.
- Extended `ComponentSpec` with residual/assembly metadata:
  - ports;
  - state names;
  - algebraic variables;
  - residual names;
  - output paths;
  - accepted map slots;
  - transient input paths;
  - optional residual contract.
- `EngineAssembler.source_catalog()` now pulls source and residual paths from
  component specs and residual contracts in addition to the previous
  compatibility paths.
- Added `EngineAssembler.residual_network_problem()`, which compiles all
  registered component-local residual contracts into a square `NetworkProblem`.
  Example 19 now proves the FFSC graph can at least assemble its component-local
  algebraics into the Phase 10 network container.
- Added unit coverage for:
  - registry residual metadata;
  - valve residual evaluation;
  - nozzle residual evaluation;
  - example 19 component-contract network assembly.

Remaining limitation:

- Phase 11 is partially complete. The implemented contracts are enough to
  establish the residual-provider interface and a square component-local
  network, but they are not yet a full FFSC solve. Missing pieces include pipe
  momentum residual contracts, chamber/preburner multi-inlet residuals,
  branch/merge residuals, true pump/turbine map residuals, units on every
  registry field, and automatic connection-port unknown generation.

### Phase 12: Solver Driver And ROCETS-Style Execution Processor

Deliverables:

- [x] Add `SolverDriver` as the numerical execution layer behind the analysis
  registry.
- [ ] Implement modes:
  - `steady`;
  - `transient`;
  - `profile`;
  - [x] `sweep`;
  - [x] `monte_carlo`;
  - `linearization`.
- [x] Add steady trim using the same residual assembly as transient execution.
- [x] Add schedule discontinuity detection and phase splitting.
- [ ] Add state activation modes during transient integration:
  - [x] active;
  - [x] inactive/fixed;
  - [x] forced steady-state.
- [ ] Add controller state vector integration for PID and other stateful
  controllers.
- [x] Add integration defaults and per-state exceptions.
- [ ] Add failed-solve recovery hooks:
  - smaller step retry;
  - algebraic warm-start reset;
  - [x] residual report;
  - optional abort criteria.

Acceptance criteria:

- [x] The public API remains `run_config_folder(path)` for all modes.
- [ ] Examples 04, 09, 10, and 13 are migrated away from custom run logic or
  explicitly marked legacy.
  - [x] Example 04 migrated to `run_config_folder`.
  - [x] Example 10 migrated to `run_config_folder`.
  - [ ] Examples 09 and 13 remain compatibility examples.
- [x] Example 19 executes a reduced-order FFSC DAE transient from YAML.
- [ ] Example 19 performs a true port-variable steady trim before transient
  integration.

Phase 12 implementation notes:

- Added `atha.runner.SolverDriver`, a thin execution layer between
  `ConfigFolderRunner` and the analysis registry.
- `ConfigFolderRunner` now loads config once and delegates execution through
  `SolverDriver`, preserving the public `run_config_folder(path)` API.
- Added `ExecutionPlan`, `ExecutionPhase`, `IntegrationOptions`, and
  `StateMode` as the driver-facing numerical execution plan.
- Added schedule breakpoint discovery for `step`, `ramp`, `table`, `profile`,
  and `runbox` schedules. The driver now builds phase boundaries from boundary
  conditions, timings, and operating targets before dispatching the analysis.
- Added parsing for:
  - `analysis.integration`;
  - `analysis.integration.per_state`;
  - `analysis.state_modes`;
  - `analysis.trim.enabled`;
  - `analysis.recovery`.
- Added `NetworkProblem.trim()` and `NetworkProblem.solve_checked()` so steady
  trim and transient algebraic solves use the same named/scaled residuals and
  raise residual-diagnostic failures through `NetworkSolveError`.
- Added unit coverage for execution-plan phase splitting, integration
  overrides, state modes, trim flags, recovery settings, and checked solve
  diagnostics.

Remaining blockers:

- The driver can now plan phase splitting and expose state modes, but the
  current compatibility solvers do not yet consume every planned phase. Full
  use of phase-by-phase integration requires the arbitrary DAE solver path to
  replace the remaining compatibility transient loops.
- Example 19 now runs a solved reduced-order FFSC DAE transient from YAML,
  including operating targets, controllers, timings, valve transients, shaft
  dynamics, chamber/preburner pressure states, algebraic residual diagnostics,
  telemetry CSV, plot, HDF5, and manifest outputs.
- True FFSC port-variable steady trim still depends on automatic port-variable
  assembly, pipe/chamber/preburner residual contracts, branch/merge residuals,
  and pump/turbine map residuals. The current implementation is a reduced-order
  acceptance solve, not the final multi-port thermodynamic solve.
- Smaller-step retry and warm-start reset recovery require the universal DAE
  transient loop to own the algebraic warm start at the driver level.
- PID/controller state integration remains blocked until controller states are
  actively integrated by the universal DAE transient loop. Phase 13 now
  registers controller state names and initial values through
  `EngineAssembler.initial_vectors()`, but compatibility solvers do not yet
  consume those states.

### Phase 13: Controller Processor Completion

Deliverables:

- [ ] Add built-in controllers:
  - [x] null/pass-through;
  - [x] proportional;
  - [x] PI;
  - [x] PID;
  - [x] scheduled gain;
  - [x] limiter;
  - [x] rate limiter;
  - [x] selector/min/max;
  - [x] splitters for total flow and OF;
  - balance-driven trim controllers.
- [x] Register controller states in the global state vector where needed.
- [ ] Add anti-windup, output limits, command rate limits, and diagnostics.
- [x] Support controller chaining with explicit signal graph validation.
- [x] Allow controller execution order to be derived from dependencies.

Acceptance criteria:

- [x] Example 18 uses a YAML-defined built-in proportional controller instead
  of `controller.py`.
- [x] Example 19 uses built-in proportional controllers for `mdot_total` and
  `OF`.
- [x] Controller telemetry includes target, measurement, error, command, and
  saturation state.

Phase 13 implementation notes:

- Added a built-in `proportional` controller type with target, measurement,
  feed-forward gain, proportional gain, and output limits.
- Added built-in controller types:
  - `pi`;
  - `pid`;
  - `scheduled_gain`;
  - `limiter`;
  - `rate_limiter`;
  - `selector`;
  - `min`;
  - `max`.
- Added dependency-derived controller execution order and cycle detection.
- Added public controller metadata helpers:
  - `controller_execution_order`;
  - `controller_input_paths`;
  - `controller_output_paths`;
  - `controller_state_infos`.
- `EngineAssembler.initial_vectors()` now includes controller states such as
  `controller.<name>.integral`, so controller state registration is available
  to the future universal DAE state vector.
- Dynamic controller output now includes diagnostic paths:
  - `controller.<name>.target`;
  - `controller.<name>.measurement`;
  - `controller.<name>.error`;
  - `controller.<name>.command`.
- Diagnostics now also include:
  - `controller.<name>.raw_command`;
  - `controller.<name>.saturated`;
  - `controller.<name>.integral`;
  - `controller.<name>.derivative`;
  - `controller.<name>.rate` where applicable.
- Example 18 now uses the built-in proportional controller and exports
  `LOX_CONTROLLER_ERROR` through telemetry.
- Example 19 now uses two built-in proportional controllers:
  - `methane_crossover_mdot_p` controls total mass flow through the methane
    crossover valve;
  - `lox_crossover_of_p` controls OF through the LOX crossover valve.
- Added unit coverage for controller dependency ordering, cycle detection,
  limiters/selectors, PI state registration, and controller states in assembler
  initial vectors.

Remaining blockers:

- PI/PID output currently uses configured initial integral/previous-error
  values in the compatibility path. Full PI/PID time evolution requires the
  universal DAE transient loop to pass and advance controller states every
  solver step.
- Anti-windup needs the same stateful controller loop plus a clear policy for
  clamping, back-calculation, or conditional integration.
- The `rate_limiter` built-in has schema, output diagnostics, and state
  registration, but true rate limiting over time is blocked until controller
  memory is advanced by the universal DAE loop.
- Balance-driven trim controllers remain blocked until user-defined algebraic
  balances are connected to `NetworkProblem` residuals.

### Phase 14: Output Processor Completion

Deliverables:

- [x] Add generic HDF5 output with:
  - [x] time histories;
  - [x] state histories;
  - [x] algebraic histories;
  - [x] residual histories;
  - [x] commands;
  - [x] controller signals;
  - [x] targets;
  - [x] boundaries;
  - [x] units;
  - [x] config provenance.
- [x] Add residual diagnostic CSV/JSON export.
- [x] Add test-data comparison reports:
  - [x] RMSE;
  - [x] max error;
  - [ ] settling time;
  - [x] overshoot;
  - [x] steady-state bias;
  - [x] channel alignment and resampling.
- [ ] Make plot generation fully telemetry-driven and independent of analysis
  type.
- [x] Add output manifests that record all generated artifacts.

Acceptance criteria:

- [ ] Examples 15-19 export through the same output processor.
- [x] HDF5 round-trip tests verify data, units, and metadata.
- [x] A telemetry typo fails before solve.

Phase 14 implementation notes:

- Added generic `write_telemetry_hdf5()` and `write_output_manifest()`.
- Examples 15-18 now export CSV, PNG, HDF5, and manifest artifacts through the
  shared telemetry writer paths.
- Added unit coverage for HDF5 and manifest export.
- Added `atha.output.OutputProcessor`, the shared orchestration layer for CSV,
  HDF5, plot, residual diagnostics, and output manifests.
- Added residual diagnostic exports:
  - `write_residual_diagnostics_csv`;
  - `write_residual_diagnostics_json`.
- Pressure-fed examples now emit residual diagnostic artifacts alongside CSV,
  HDF5, plot, and manifest outputs.
- Extended HDF5 writing with optional groups:
  - `states`;
  - `algebraics`;
  - `residuals`;
  - `boundaries`.
- Added comparison utilities:
  - `compare_time_series`;
  - `write_comparison_report_json`.
  These currently compute RMSE, max error, overshoot, steady-state bias, and
  perform channel alignment/resampling by interpolation.
- Added unit coverage for residual diagnostics, comparison report generation,
  and HDF5 state/residual groups.

Remaining blockers:

- Full state/algebraic/residual HDF5 export is structurally supported, but the
  current compatibility runners only expose partial histories. Complete
  histories for every component state, algebraic variable, residual, command,
  target, and boundary require the universal DAE solver to own `X`, `Z`, `Rz`,
  command, target, and boundary histories centrally.
- Example 19 now exports solved time histories through the reduced-order FFSC
  DAE runner. The remaining limitation is that those histories are reduced
  model states/algebraics, not yet full ROCETS-style per-port thermodynamic
  property histories.
- Settling-time comparison metrics are now available when callers provide a
  per-channel or scalar settling tolerance. The utility intentionally does not
  guess settling bands without tolerances.
- Plot generation is telemetry-driven, but still called from compatibility
  analysis handlers through `OutputProcessor`. It becomes fully independent of
  analysis type once all analyses return a common sampled-output object.

### Phase 15: Linearization And Reduced-Order Modes

Deliverables:

- [x] Add finite-difference linearization around a trimmed operating point.
- [ ] Add optional analytic partial hooks for components.
- [x] Export state-space matrices:
  - [x] `A`;
  - [x] `B`;
  - [x] `C`;
  - [x] `D`;
  - [x] state/input/output labels;
  - [x] operating point.
- [ ] Support ROCETS-style state forcing to steady-state for reduced-order
  studies.
- [x] Add perturbation size defaults plus per-state/per-input exceptions.

Acceptance criteria:

- [x] A simple pressure-fed TCA linearizes around steady trim.
- [x] Example 19 can generate a linearization artifact at the 40 kg/s, OF 3.4
  operating point after the FFSC DAE solve is implemented.
- [x] Linearization output is independent of plot/telemetry output formatting.

Phase 15 implementation notes:

- Added `atha.analysis.linearization`, with:
  - `finite_difference_state_space`;
  - `StateSpaceLinearization`;
  - `PerturbationConfig`;
  - `write_linearization_json`.
- Added pressure-fed TCA linearization around a steady trim through
  `linearize_pressure_fed_tca`. This uses the same pressure-fed algebraic
  residual assembly as transient execution and exports a standalone
  linearization JSON artifact.
- Added reduced FFSC linearization for example 19. The artifact is written to
  `outputs/ffsc_dae_acceptance.linearization.json` and includes `A`, `B`, `C`,
  `D`, labels, operating point, and perturbation sizes.
- Added YAML controls under `analysis.linearization` for example 19:
  - `enabled`;
  - `output`;
  - perturbation defaults and overrides.
- Added unit coverage proving:
  - example 19 emits a linearization artifact;
  - example 18/pressure-fed TCA linearizes around steady trim;
  - generated matrices are finite and dimensionally consistent.

Previously blocked deliverables revisited:

- Phase 12 `linearization` mode is partially unblocked: the numerical
  linearization primitive and model adapters exist, but the analysis registry
  still does not have a standalone `analysis.type: linearization` handler.
  That remains blocked on deciding whether linearization is a primary analysis
  type or a post-processor attached to transient/steady analyses.
- Phase 12 controller state integration remains blocked. PI/PID state names are
  registered, but the compatibility solvers still do not integrate controller
  state derivatives in a universal DAE loop.
- ROCETS-style state forcing remains only represented in execution-plan
  metadata. The linearization code can linearize a reduced model at an
  operating point, but it does not yet force selected dynamic states to
  algebraic steady-state inside the solver.
- Phase 14 full state/algebraic/residual output is partly unblocked for example
  19, which now exports reduced state, algebraic, and residual histories. It is
  still blocked for arbitrary networks until the full port DAE solver owns all
  `X`, `Z`, and `Rz` histories centrally.
- Phase 17 linearization validation is partially unblocked because example 19
  now emits a machine-readable linearization artifact and its presence is
  checked by the acceptance report. Golden eigenvalue checks are not
  implemented yet because eigenvalue tolerances need to be chosen against a
  stable port-solve formulation.

Remaining blockers:

- Optional analytic partial hooks are not implemented. They should be added
  after the component residual-provider interface stabilizes in the full
  arbitrary port solver; adding hooks against the reduced FFSC model would
  create throwaway interfaces.
- Example 19 linearization is reduced-order, not full port-variable. The full
  ROCETS-like linearization requires Phase 20 automatic port unknowns,
  connection residuals, component residual contracts, and thermodynamic state
  propagation.

### Phase 16: Sweep And Monte Carlo Runner Integration

Deliverables:

- [x] Move sweep and Monte Carlo orchestration behind the analysis registry.
- [x] Support parameter perturbations without editing engine YAML.
- [ ] Support structured sweeps over:
  - [ ] component parameters;
  - [x] map scale factors;
  - [x] boundary conditions;
  - [ ] controller gains;
  - [ ] initial conditions.
- [x] Support random sampling distributions and seed control.
- [ ] Export ensemble statistics and sensitivity metrics through the output
  processor.

Acceptance criteria:

- [x] Examples 04 and 10 run from `run_config_folder`.
- [x] Existing Monte Carlo unit tests remain valid through the registry path.
- [ ] Sweep outputs include the varied parameter values and output metrics.

Phase 16 implementation notes:

- Added `atha.analysis.gg_mc_sweep.run_nominal_mc_sweep`, a registry-backed
  gas-generator nominal, Monte Carlo, and speed-sweep runner.
- Registered `analysis.type: nominal_mc_sweep` in the analysis registry with
  mode `sweep`.
- Refactored examples 04 and 10 so their `run.py` files call
  `run_config_folder(configs)` instead of owning the solver, Monte Carlo, and
  sweep orchestration.
- The shared runner supports:
  - nominal steady trim;
  - random Monte Carlo with existing LHS/Saltelli infrastructure, distribution
    definitions, and seed control;
  - parameter perturbations for pump map efficiency scale factors, gas
    generator efficiency, and inlet boundary temperatures without editing
    engine YAML;
  - structured speed sweeps through the shaft speed boundary override;
  - HDF5 Monte Carlo result export and histogram/sweep plots.
- Added registry unit coverage for the new `nominal_mc_sweep` type.
- Verified:

  ```powershell
  .venv\Scripts\python.exe examples\04_gg_single_shaft_mc_sweep\run.py
  .venv\Scripts\python.exe examples\10_gg_lox_methane\run.py
  .venv\Scripts\python.exe -m pytest tests\unit\test_config.py tests\unit\test_solver.py -q
  ```

Previously blocked deliverables revisited:

- Phase 12 sweep/Monte Carlo modes are now partially unblocked through the
  analysis registry for examples 04 and 10. `profile`, standalone `steady`, and
  standalone `linearization` registry modes remain open.
- Phase 12 migration is improved: examples 04 and 10 are now migrated away from
  custom orchestration. Examples 09 and 13 still remain compatibility examples.
- Phase 16 parameter perturbations are available for the gas-generator runner,
  but not yet generalized as a schema-level patch system that can target any
  YAML path in any analysis.

Remaining blockers:

- Structured sweeps over arbitrary component parameters, controller gains, and
  initial conditions are not implemented. This needs a generic config
  perturbation layer that can patch loaded YAML objects by path before model
  construction.
- Ensemble statistics are exported through the existing Monte Carlo HDF5 file,
  but not yet through `OutputProcessor`. `OutputProcessor` currently assumes
  time-series telemetry; ensemble tables need a sibling artifact path.
- Sensitivity metrics are not computed in the registry-backed runner yet. The
  existing Monte Carlo package has Sobol storage support, but Phase 16 did not
  wire a sensitivity-analysis execution mode.

### Phase 17: Validation Matrix And Acceptance Tests

Deliverables:

- [x] Define a validation matrix matching ROCETS-style expectations:
  - [x] component-level verification;
  - [x] subsystem verification;
  - [x] full engine steady trim;
  - [x] throttle transient;
  - [x] start transient;
  - [ ] shutdown transient;
  - [x] closed-loop controller transient;
  - [x] linearization.
- [ ] Add golden regression outputs for examples 15-18.
- [x] Add acceptance tolerances for example 19:
  - [x] steady/reduced endpoint closure;
  - [x] mass-flow target tracking;
  - [x] OF tracking;
  - [ ] valve command response;
  - [x] shaft speed response;
  - [ ] pump/turbine map response;
  - [x] thrust near 150 kN at 40 kg/s.
- [x] Add comparison utilities for external test/simulation data.
- [x] Add CI-friendly runtime tiers:
  - [x] unit;
  - [x] fast integration;
  - [x] slow acceptance;
  - [x] Monte Carlo.

Acceptance criteria:

- [x] Example 19 is the first solved FFSC DAE acceptance gate.
- [ ] Example 19 is the first full port-variable FFSC DAE acceptance gate.
- [x] A failing acceptance run reports whether the failure is numerical,
  physical-model, controller, or telemetry related.
- [x] Validation outputs are written as machine-readable artifacts and human
  review plots.

Phase 17 implementation notes:

- Example 19 now runs through `run_config_folder()` as a solved reduced-order
  FFSC DAE acceptance transient instead of a configuration-only validator.
- The runner exports CSV, plot, HDF5, manifest, and residual diagnostic
  artifacts through the shared output processor.
- Unit coverage now checks that example 19 solves, exports artifacts, returns
  near-40 kg/s final mass flow, and returns near-150 kN final thrust.
- Added `development/validation_matrix.md`, which records the ROCETS-style
  validation ladder from component checks through subsystem, full-engine,
  closed-loop, and linearization gates.
- Added `atha.validation.acceptance`, with machine-readable acceptance reports
  and categorized checks:
  - `numerical`;
  - `physical_model`;
  - `controller`;
  - `telemetry`;
  - `linearization`.
- Example 19 now writes
  `outputs/ffsc_dae_acceptance.acceptance.json` and prints PASS/FAIL from that
  report.
- Added explicit example 19 acceptance tolerances in
  `examples/19_ffsc_dae_acceptance/configs/analysis.yaml`.
- Added optional settling-time metrics to `compare_time_series` when callers
  provide settling tolerances.
- Added pytest marker tiers in `pyproject.toml`:
  - `fast`;
  - `integration`;
  - `acceptance`;
  - `monte_carlo`;
  - `slow`.

Remaining blockers:

- This is not yet the final arbitrary ROCETS-like port solve. The reduced-order
  FFSC runner computes pump-capacity branch flows, preburner/chamber pressure
  states, shaft dynamics, nozzle flow, thrust, and named algebraic residuals.
  Full completion still requires automatic fluid-port unknown generation,
  connection pressure/enthalpy/mass residuals, pipe inertia residual contracts,
  chamber/preburner thermochemistry residuals, and map-backed pump/turbine
  residuals.
- Golden regression outputs for examples 15-18 are not implemented yet. The
  examples are being re-run as smoke/regression checks, but writing fixed
  golden files now would lock in compatibility-runner behavior before the full
  port solver lands.
- Valve command response and pump/turbine map response are not yet separate
  example 19 acceptance checks. The reduced model validates shaft response and
  target tracking, but true map-response validation needs map-backed component
  residuals and per-port histories from Phase 20.
- Shutdown transient validation is not implemented because there is no shutdown
  profile example or acceptance target yet.

### Phase 18: Documentation, CLI, And User Workflow

Deliverables:

- [x] Add `atha-run <config-folder>` CLI.
- [x] Document YAML schemas with examples for each file type.
- [x] Document path grammar:
  - [x] `component.port.variable`;
  - [x] `component.state`;
  - [x] `component.output`;
  - [x] `commands.*`;
  - [x] `targets.*`;
  - [x] `timings.*`;
  - [x] `boundaries.*`;
  - [x] `residuals.*`.
- [x] Document how to add a component, map, transient, controller, and
  telemetry channel.
- [x] Document solver diagnostics and common convergence failures.
- [x] Update the README to distinguish:
  - [x] runnable examples;
  - [x] validation examples;
  - [x] legacy/compatibility examples;
  - [x] acceptance examples.

Acceptance criteria:

- [x] A new user can create a pressure-fed TCA case from YAML only.
- [x] A contributor can add a new component residual provider without editing
  the runner.
- [x] Example 19 README explains what is validated today and what is pending.

Phase 18 implementation notes:

- Added `atha.cli:main` and the `atha-run` project script entrypoint.
- Verified the CLI with example 18:

  ```powershell
  .venv\Scripts\python.exe -m atha.cli examples\18_tca_mdot_controller\configs
  ```
- Extended CLI output to print linearization, acceptance, Monte Carlo, and
  sweep artifacts when present.
- Added documentation:
  - `docs/configuration.md`;
  - `docs/path_grammar.md`;
  - `docs/contributing_models.md`;
  - `docs/solver_diagnostics.md`.
- Rewrote `README.md` to reflect the current example set and distinguish:
  - runnable registry examples;
  - acceptance examples;
  - compatibility examples.
- Updated `examples/19_ffsc_dae_acceptance/README.md` so it describes the
  current solved reduced-order acceptance case instead of the earlier
  config-only validator.

Previously blocked deliverables revisited:

- Phase 14 diagnostic documentation is now unblocked: HDF5 groups, residual
  diagnostics, manifests, linearization, and acceptance outputs are documented
  in `docs/solver_diagnostics.md`.
- Phase 17 validation workflow documentation is now unblocked:
  `development/validation_matrix.md` is linked from README and acceptance
  report categories are documented.
- Phase 12/16 user workflow is clearer: examples 04 and 10 are documented as
  registry-backed Monte Carlo/sweep examples, while examples 09 and 13 are
  explicitly listed as compatibility workflows.

Remaining blockers:

- The docs describe the current reduced-order and compatibility runner
  behavior. They cannot fully document the arbitrary ROCETS-like port solver
  until Phase 20 implements automatic port unknowns, connection residuals, and
  universal state/algebraic histories.
- A contributor can add component-local residual providers without editing the
  runner, but full port-coupled component contracts still require Phase 20
  interfaces.

### Phase 19: Runner Extension Contracts And Perturbation Foundation

Deliverables:

- [x] Add a shared analysis execution context for registry handlers.
- [x] Preserve existing `(config_path, output_dir)` handlers while allowing
  future context-aware handlers.
- [x] Standardize generated artifact discovery in `RunResult`.
- [x] Improve unsupported `analysis.type` errors with known-type listings,
  close-match suggestions, and contributor guidance.
- [x] Add a generic dotted-path override primitive for loaded config objects.
- [x] Document how to add a new analysis type.
- [ ] Wire generic dotted-path overrides into all sweep and Monte Carlo YAML
  schemas.
- [ ] Convert every analysis handler to the context-aware signature.
- [ ] Make `analysis.type: steady`, `analysis.type: transient`, and
  `analysis.type: linearization` fully generic aliases over the port solver.

Acceptance criteria:

- [x] Public runner results expose CSV, plot, HDF5, manifest, residual,
  linearization, acceptance, Monte Carlo, and sweep artifacts through one
  object.
- [x] A misspelled analysis type reports actionable suggestions.
- [x] Config perturbations can target nested dataclasses, dictionaries, and
  list entries without mutating the original loaded config.
- [x] Examples 15-19 continue to run through the public config runner.

Phase 19 implementation notes:

- Added `atha.runner.AnalysisContext`. `SolverDriver` now builds this context
  after loading configs and execution-plan metadata, attaches it to summaries
  that support dynamic attributes, and dispatches through
  `AnalysisRegistry.run_context()`.
- Added `RunArtifacts` and surfaced it through `RunResult.artifacts`,
  `RunResult.artifact_paths()`, and runner metadata. This gives CLI, examples,
  and tests one stable artifact contract instead of probing summary attributes.
- Extended `AnalysisRegistry` with close-match diagnostics for unsupported
  analysis types and a compatibility path for context-aware handlers.
- Added `atha.config.apply_path_overrides()` and `flatten_overrides()` as the
  schema-level perturbation primitive needed for arbitrary sweeps and Monte
  Carlo studies.
- Updated `docs/contributing_models.md` with the analysis-registry extension
  workflow.

Previously blocked deliverables revisited:

- Phase 12 unsupported-analysis diagnostics are now unblocked. The registry
  reports known analysis types and close matches before any solver work begins.
- Phase 16 arbitrary perturbation support is partially unblocked. A general
  loaded-config patch primitive exists, but the gas-generator Monte Carlo and
  sweep runner still uses its current domain-specific perturbation schema.
- Phase 14 artifact discovery is unblocked for public runner callers. Full
  universal output still depends on every analysis returning common histories
  from the same port-solver object.
- Phase 18 contributor workflow is improved with documented registry extension
  instructions.

Remaining blockers:

- Generic dotted-path perturbations are not yet connected to every analysis
  mode. The next step is adding a normalized `analysis.perturbations` schema
  that calls `apply_path_overrides()` before model construction, then replacing
  domain-specific sweep patch code case by case.
- Most registered handlers still use the legacy `(config_path, output_dir)`
  signature. They are compatible with the Phase 19 context bridge, but full
  migration should wait until the port solver owns common histories and source
  catalogs; otherwise handlers would still reload or reinterpret pieces of the
  same config internally.
- Generic `steady`, `transient`, and `linearization` aliases remain blocked by
  the same full arbitrary ROCETS-like port solve work: automatic port unknown
  generation, connection residual assembly, thermodynamic propagation, and
  universal state/algebraic history ownership.

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
