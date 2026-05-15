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

Post-Phase-24 update:

- Generic `steady`, `profile`, and `linearization` modes now exist through the
  analysis registry and `DAEExecutionProblem`.

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
  from 20-30 s.
- `OF`: 3.0 through 5 s, ramping to 3.4 at 25 s.

Controllers:

- Total mass flow target drives `methane_crossover_valve.command` with a
  proportional controller, initial crossover valve position 50%.
- OF target drives `lox_crossover_valve.command` with a proportional
  controller, initial crossover valve position 50%.
- `main_lox_valve` and `main_methane_valve` are fixed at 100% command.

Remaining implementation actions for this acceptance case:

- [x] Extend the analysis registry so `analysis.type: ffsc_dae_transient` is
  dispatched through the public runner.
- [x] Build a global DAE unknown catalog from arbitrary `engine.yaml`
  connections:
  - [x] port pressures;
  - [x] port mass flows;
  - [x] temperature or enthalpy;
  - [x] shaft speed and torque variables where algebraic coupling is required.
- [ ] Add component residual providers and derivative hooks for:
  - `Pump` map pressure rise and shaft torque load are partly implemented;
    efficiency/output enthalpy and full flow compatibility remain open;
  - `FlowSplitter` mass conservation and branch split residuals;
  - `Pipe` quasi-steady friction residuals and first-order `mdot` state
    dynamics are implemented; physical inertance and compressible storage
    remain open;
  - `Valve` flow residuals using transient valve position;
  - `MassFlowInjector` pressure-drop and flow residuals;
  - `Preburner` mass, mixture-ratio, pressure, temperature residuals, and
    pressure-state dynamics are partly implemented; energy and outlet state
    remain open;
  - `Turbine` pressure ratio, efficiency, power, and shaft torque drive are
    partly implemented; outlet state remains open;
  - `Rotor` speed dynamics coupled to pump/turbine power are implemented in
    the universal DAE loop;
  - `CombustionChamber` mass/pressure/mixture residuals and pressure-state
    dynamics are partly implemented; energy and outlet state remain open;
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
  - [x] example 19 runs from `run_config_folder`;
  - [x] `mdot_total` follows the 40 -> 30 -> 40 kg/s target profile in the
    reduced acceptance model;
  - [x] OF ramps from 3.0 to 3.4 in the reduced acceptance model;
  - [x] controlled valve commands move in response to target error;
  - [x] preburner flow changes alter turbine torque in the reduced acceptance
    model;
  - [x] shaft speeds respond dynamically;
  - [x] pump pressure rise changes with shaft speed through loaded map
    interpolation;
  - [x] chamber/nozzle thrust settles near 150 kN during 40 kg/s operation in
    the reduced acceptance model.

### Phase 7: Analysis Registry

Deliverables:

- [x] `steady` diagnostics through `port_network_diagnostics`;
- [x] `transient`;
- [x] `profile`;
- [x] `sweep` through `nominal_mc_sweep`;
- [x] `monte_carlo` through `nominal_mc_sweep`;
- [x] `linearization`;

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
  - `ffsc_dae_transient`;
  - `nominal_mc_sweep`;
  - `port_network_diagnostics`.
- Added `analysis_mode` metadata to `RunResult`.
- Added `atha.analysis.ffsc_acceptance.validate_ffsc_dae_acceptance`, allowing
  example 19 to execute through `run_config_folder` as a configuration and
  schedule validation acceptance case while the full arbitrary FFSC DAE solve is
  still pending.
- Refactored `examples/19_ffsc_dae_acceptance/run.py` to use the public runner.
- Added unit coverage for registered analysis types and public-runner dispatch.

Post-Phase-20 update:

- The registry now supports transient analyses, sweep/Monte Carlo through the
  gas-generator runner, and steady algebraic diagnostics through the generic
  port-network diagnostic handler.

Post-Phase-24 update:

- The registry now includes generic `steady`, `profile`, and `linearization`
  analysis types backed by `DAEExecutionProblem`.

Remaining limitation:

- A physically complete generic `steady` solve remains blocked by the same
  component residual and thermodynamic closure gaps documented in Phase 20-23.

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

- `NetworkProblem` and `PortNetworkBuilder` now provide a generic algebraic
  `Z/Rz` foundation, but the current solved transients still use
  compatibility-specific RHS functions.
- There is no universal execution object that owns:
  - state vector `X`;
  - algebraic vector `Z`;
  - residual vector `Rz`;
  - command vector `U`;
  - output/source catalog;
  - residual scales and diagnostics;
  - state, algebraic, command, target, boundary, and residual histories.
- Connection residuals are now generated from arbitrary `engine.yaml`
  connections, but arbitrary layouts are not yet solved by a warm-started
  algebraic loop inside a universal transient integrator.
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
- [x] Add reusable YAML library support for:
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

- [x] Add integration option defaults plus per-state exceptions.
- [x] Add state mode declarations:
  - [x] `active`;
  - [x] `inactive`;
  - [x] `fixed`;
  - [x] `steady_state`.
- [x] Validate controller outputs against legal command or transient input
  paths.

Acceptance criteria:

- [x] Invalid YAML keys fail before solve with actionable messages.
- [ ] Example 19 can declare algebraic balances for `mdot_total` and `OF`
  without Python changes.
- [x] A config folder can reuse a valve template or telemetry channel group
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
- Added pre-schema YAML include expansion through `include` and `$include`.
  Includes are resolved relative to the declaring YAML file, can be a single
  path or ordered list of paths, and are expanded before strict dataclass
  validation. Merge policy:
  - mappings are deep-merged;
  - local scalar values override included scalar values;
  - lists are appended for sections such as `connections`, `channels`, and
    `events`;
  - include cycles raise `ConfigError`.
- Added unit coverage proving YAML includes can reuse component fragments and
  telemetry channel groups, and that include cycles fail clearly.

Remaining limitation:

- Phase 8 is partially complete. Reusable YAML libraries are implemented at the
  file-merge level. User-defined algebraic balances still need implementation.

Additional implementation notes:

- Built-in proportional controllers are now supported and validated. This
  removed the need for example 18 to load `controller.py` for the simple
  mass-flow control case.

Remaining blockers:

- Higher-level library packaging remains open for named template catalogs and
  parameterized subsystem fragments. The lower-level include/merge mechanism is
  implemented and provides the precedence policy those features can build on.
- User-defined algebraic balances are still open because the expression parser
  and safe variable binding need to be tied to the Phase 10/11 network residual
  system. String expressions should not be evaluated ad hoc.
- Per-state integration exceptions and state modes are parsed into
  `ExecutionPlan`; full enforcement remains blocked until the universal DAE
  loop owns all physical, transient, and controller states.

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
- [x] Example 19 produces a complete source catalog, including shaft speeds,
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
- Post-Phase-20 update: `EngineAssembler.port_network_problem()` now builds
  automatic port variables and residuals from the engine graph. This unblocks
  topology diagnostics, but it does not replace compatibility model extraction
  for the pressure-fed and reduced FFSC transient solvers yet.

### Phase 10: Generic Global DAE Network Problem

Deliverables:

- [x] Add `atha.network.NetworkProblem`.
- [x] Build a global algebraic unknown vector `Z` from connected port variables:
  - [x] pressure;
  - [x] mass flow;
  - [x] enthalpy or temperature;
  - [ ] density where required;
  - [x] shaft torque/speed algebraics;
  - [x] thermal heat-flow algebraics.
- [x] Build a named/scaled global residual vector `Rz` for registered residual
  equations.
- [x] Support warm-started algebraic solves at each transient RHS call.
- [x] Detect non-square systems and report missing or over-specified residuals.
- [x] Detect algebraic loops through global residual assembly and solve them
  through root or least-squares methods rather than relying on
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

Post-Phase-20 update:

- Arbitrary connected port variables, shaft/thermal algebraics, and automatic
  component residual assembly now exist through `PortNetworkBuilder`.

Remaining limitation:

- Phase 10 is structurally in place, but the full ROCETS-like DAE network is
  not complete until component residual contracts cover pipe, chamber,
  preburner, map-backed pump/turbine behavior, fluid properties, and dynamic
  state derivatives. The automatic network can diagnose and solve anchored
  algebraic systems, but it does not yet own the universal transient RHS.

### Phase 11: Component Residual Contracts

Deliverables:

- [x] Define a residual-provider interface for all reusable components.
- [ ] Implement residual contracts for:
  - [x] `Valve`;
  - [x] `Pipe`;
  - [x] `MassFlowInjector`;
  - [x] `FlowSplitter`;
  - [x] `CombustionChamber`;
  - [x] `Preburner`;
  - [x] `Nozzle`;
  - [x] `Pump`;
  - [x] `Turbine`;
  - [x] `Rotor`;
  - [x] `RegenChannel`;
  - [x] thermal nodes through the initial regen/thermal contract.
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
- [x] Make pump maps support corrected speed, corrected flow, head or pressure
  rise, and optional extrapolation rules through the loaded map object.
- [x] Make turbine maps support pressure ratio, corrected flow, efficiency, and
  power residuals.
- [x] Add branch residuals for `FlowSplitter` outlet flow splits.
- [ ] Add merge residuals for preburner/chamber multi-inlet mixing.

Acceptance criteria:

- [x] Example 19 component graph assembles into a square residual system for
  registered component-local residual contracts.
- [x] Pump and turbine map values change with shaft speed and flow.
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
- Post-Phase-22 update: added:
  - `PipeMomentumContract`;
  - `FiniteVolumeCombustorContract` for `CombustionChamber` and `Preburner`;
  - `RegenThermalContract`.
- Post-Phase-23 update:
  - `PumpHeadContract` can evaluate attached `head_map` outputs such as
    `pressure_rise`, `head`, or `delta_P` using speed and flow ratios;
  - `TurbinePowerContract` can evaluate attached efficiency maps using pressure
    ratio and corrected-flow ratio.
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
  - pipe residual evaluation;
  - combustor residual evaluation;
  - example 19 component-contract network assembly.

Remaining limitation:

- Phase 11 is partially complete. Pipe momentum, chamber/preburner first-pass
  residuals, regen heat residuals, and map-backed pump/turbine scalar
  residuals now exist, but they are not yet a full FFSC solve. Missing pieces
  include general merge residuals, high-fidelity thermodynamic state closure,
  units on every registry field, and full dynamic coupling into the universal
  DAE loop.

### Phase 12: Solver Driver And ROCETS-Style Execution Processor

Deliverables:

- [x] Add `SolverDriver` as the numerical execution layer behind the analysis
  registry.
- [x] Implement modes:
  - [x] `steady`;
  - `transient`;
  - [x] `profile`;
  - [x] `sweep`;
  - [x] `monte_carlo`;
  - [x] `linearization`.
- [x] Add steady trim using the same residual assembly as transient execution.
- [x] Add schedule discontinuity detection and phase splitting.
- [x] Add state activation modes during transient integration:
  - [x] active;
  - [x] inactive/fixed;
  - [x] forced steady-state.
- [x] Add controller state vector integration for PID and other stateful
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
- Added support for explicit named phases under `analysis.time.phases`, for
  example `startup`, `CLC`, and `shutdown`. When explicit phases are present,
  they are preserved in `ExecutionPlan.phases` with names so controller
  activation can be tied to simulation phase.
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
- PID/controller state integration is implemented for the universal DAE loop
  and sampled compatibility controllers now share the same derivative
  convention. Remaining work is to migrate all compatibility solvers onto the
  universal DAE execution object.

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
- [x] Add anti-windup, output limits, command rate limits, and diagnostics in
  the generic DAE execution path.
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
- PID derivative no longer uses `controller.<name>.previous_error` as an ODE
  state. Previous error is sampled from the controller hold cache and divided
  by the controller evaluation period, so derivative gain has the expected
  `d(error)/dt` semantics.
- Sampled compatibility runners now carry PI/PID integral memory through the
  controller hold cache. The integral is advanced by `error * dt` when the
  controller is active, and the default anti-windup policy freezes integration
  when the previous sampled command was saturated.
- Controller diagnostics now expose explicit command contributions:
  - `controller.<name>.proportional_term`;
  - `controller.<name>.integral_term`;
  - `controller.<name>.derivative_term`.
- `active_phases` is now supported on controller blocks. Controllers with this
  field only emit outputs when the current named phase matches; examples 19 and
  20 use `active_phases: [CLC]` so startup and shutdown remain open-loop.
- Controller config validation now accepts `active_phases` as a common field on
  all built-in controller types.
- Compatibility runners for examples 19 and 20 now pass the current phase,
  controller `dt`, and previous sampled controller diagnostics into the shared
  controller evaluator.
- Compatibility runners fill missing transient command paths from each
  transient's configured initial value, so inactive phase-scoped controllers do
  not leave the transient system or telemetry exporter without a command path.
- Examples 19 and 20 now use aligned named phases: `startup` from `0-3 s`,
  `CLC` from `3-25 s`, and `shutdown` from `25-30 s`.
- Shutdown timing events at `t = 25 s` now close the open-loop setpoints for
  crossover valves in example 19 and gas-generator valves in example 20. With
  `active_phases: [CLC]`, feedback controllers stop emitting during shutdown
  and the timing commands own the closing setpoints.
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

- PI/PID derivative behavior is fixed for both the universal DAE loop and the
  sampled compatibility runners. Integral evolution is still only fully
  stateful in the universal DAE loop; reduced compatibility runners continue
  to use the configured initial integral unless they migrate to
  `DAEExecutionProblem`.
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

### Phase 20: Automatic Port-Network Solve Foundation

Deliverables:

- [x] Generate algebraic port unknowns automatically from arbitrary
  `engine.yaml` connections.
- [x] Support fluid, shaft, and thermal connection domains in the unknown and
  residual catalog.
- [x] Assemble connection residuals:
  - [x] fluid pressure continuity;
  - [x] fluid enthalpy continuity;
  - [x] fluid mass-flow continuity;
  - [x] shaft speed continuity;
  - [x] shaft torque balance;
  - [x] thermal temperature continuity;
  - [x] thermal heat-flow balance.
- [x] Integrate registered component residual contracts into the automatic
  port network.
- [x] Add boundary-condition anchoring residuals for direct boundary paths.
- [x] Allow non-square algebraic networks to solve through least squares for
  diagnostics instead of failing during construction.
- [x] Expose port variables and residuals through the generic source catalog.
- [x] Add a registry-backed `analysis.type: port_network_diagnostics` handler.
- [x] Add tests proving:
  - [x] example 19 generates automatic FFSC port variables and connection
    residuals;
  - [x] an anchored arbitrary fluid connection solves through the port network;
  - [x] the port-network diagnostic analysis runs through `run_config_folder`.
- [ ] Replace the reduced-order example 19 FFSC runner with the automatic
  port-network transient DAE loop.
- [ ] Implement full thermodynamic property propagation for all fluid ports.
- [ ] Implement map-backed pump and turbine residuals using actual loaded map
  interpolation instead of reduced affinity approximations.
- [ ] Implement dynamic pipe inertance/friction residuals as first-class
  component states coupled to the port network.
- [ ] Implement chamber/preburner finite-volume thermochemistry residuals.
- [ ] Implement universal controller-state integration inside the DAE state
  vector.

Acceptance criteria:

- [x] A YAML-only topology can be converted to a named port-variable network
  without analysis-specific Python code.
- [x] The network reports all generated variable and residual names in a
  machine-readable diagnostic artifact.
- [x] Example 19 can be inspected as an automatic port network, including pump,
  valve, splitter, injector, chamber/nozzle, turbine, shaft, and connection
  residual names.
- [ ] Example 19 is solved by the automatic port network rather than the
  reduced FFSC algebraic model.
- [ ] Arbitrary FFSC transients close all port pressure, mass-flow, enthalpy,
  shaft torque, and finite-volume residuals with physically meaningful
  component models.

Phase 20 implementation notes:

- Added `atha.network.ports` with:
  - `Port`;
  - `PortNetworkCatalog`;
  - `PortNetworkBuilder`.
- `PortNetworkBuilder` scans `engine.yaml` connections and creates:
  - fluid port variables: `<component>.<port>.P`,
    `<component>.<port>.mdot`, `<component>.<port>.h`;
  - shaft port variables: `<component>.<port>.omega`,
    `<component>.<port>.tau`;
  - thermal port variables: `<component>.<port>.T`,
    `<component>.<port>.Q_dot`.
- The builder adds connection residuals for continuity or balance by domain and
  merges registered component residual contracts, including current Valve,
  MassFlowInjector, FlowSplitter, Nozzle, Pump, Turbine, and Rotor contracts.
- Added generic coupling residuals that connect component-local algebraic
  outputs such as `component.mdot`, `component.delta_P`, and `component.omega`
  to matching port variables where enough topology exists.
- Boundary conditions now anchor matching generated algebraic variables through
  named residuals such as `source.outlet.P_boundary_residual`.
- `NetworkProblem` now supports non-square systems when `require_square=False`.
  Square systems still try a root solve first; non-square or failed square
  systems fall back to scaled least squares. This is required for diagnostics
  because partially specified arbitrary topologies often have more equations
  than unknowns or missing closure equations during model development.
- Added `EngineAssembler.port_network_problem()` and included automatic port
  variable/residual paths in the generic telemetry/source catalog.
- Added `atha.analysis.port_network.run_port_network_diagnostics` and
  registered `analysis.type: port_network_diagnostics`.

Previously blocked deliverables revisited:

- Phase 6 automatic port unknown generation and connection residual assembly
  are now partially unblocked. The implementation is generic over topology and
  domain, but not yet tied to a universal transient RHS.
- Phase 12 arbitrary topology diagnostics are improved. A user can now build
  and inspect a named algebraic network directly from YAML without writing an
  analysis-specific model builder.
- Phase 14 source catalog completeness is improved because generated port
  variables and residuals are now discoverable paths.
- Phase 17 example 19 validation is improved at the structure level: the FFSC
  acceptance engine can now produce automatic port-network variables and
  residuals. The acceptance solve itself is still reduced-order.

Remaining blockers:

- This is not yet the full arbitrary ROCETS-like transient DAE solve. The
  missing center is a universal DAE loop that owns `X`, `Z`, `dXdt`, `Rz`,
  controller states, transient states, warm-started algebraic solves, and
  output histories for all analyses. The first universal loop now exists, but
  high-fidelity component closures and migration of legacy examples remain
  incomplete.
- Component residual contracts are still simplified. Pump and turbine contracts
  now consume loaded map objects for first-pass pressure-rise/power behavior,
  but they do not yet provide full corrected-property, efficiency, enthalpy,
  outlet-state, and energy-balance residual closure.
- Fluid thermodynamics are path placeholders. Port enthalpy variables exist and
  continuity residuals are assembled, but ATHA does not yet compute fluid
  state from species, phase, pressure, enthalpy, density, temperature, gamma,
  or mixture properties.
- Pipe dynamics are partly present through generic first-order `Pipe.mdot`
  state derivatives, but they are not yet physical inertance/compressibility
  port residual providers. The arbitrary port network still needs inertance,
  compressibility, and optional volume storage equations.
- Preburner and chamber models are not full thermochemical finite-volume
  residual providers. They need mass, species/mixture, energy, pressure, and
  outlet state equations before an arbitrary FFSC engine can be solved from
  ports alone.
- Rotor dynamics are now present in the universal DAE loop as shaft
  differential states coupled to pump load and turbine drive power. Remaining
  work is to improve the pump/turbine power, efficiency, and enthalpy closures
  that feed those rotor derivatives.
- Boundary anchoring now supports evaluated time-varying schedules through the
  universal DAE loop, but external source/sink topology is still incomplete for
  arbitrary pump inlets, tank ports, turbine exhausts, and ambient outlets.
- The `port_network_diagnostics` analysis is intentionally diagnostic. It can
  solve anchored algebraic networks and report residual closure, but it is not
  a substitute for a physically complete transient FFSC solve.

### Phase 21: Universal DAE Execution Loop

Purpose:

Replace compatibility transient RHS functions with one execution loop that
evaluates schedules, controllers, transients, states, port algebraics, residuals,
and telemetry histories through shared objects.

Deliverables:

- [x] Add a `DAEExecutionProblem` that owns:
  - [x] state vector `X`;
  - [x] algebraic vector `Z`;
  - [x] derivative vector `dXdt`;
  - [x] residual vector `Rz`;
  - [x] command/target/boundary/measurement dictionaries;
  - [x] warm-started algebraic state;
  - [x] full state/algebraic/residual histories.
- [x] Connect `PortNetworkBuilder` to the transient RHS so every solver step
  closes `Rz(t, X, Z, U)`.
- [x] Evaluate boundary, timing, operating-condition, controller, and transient
  schedules through `SolverDriver` phase boundaries.
- [x] Apply `analysis.state_modes` during integration, including fixed and
  forced-steady states.
- [x] Integrate controller states for PI/PID/rate-limiter blocks.
- [x] Integrate generic component-owned dynamic states for:
  - [x] pipe first-order mass-flow lag states;
  - [x] combustor/preburner pressure storage states;
  - [x] rotor speed states driven by pump/turbine power balance.
- [ ] Implement failed-solve recovery:
  - smaller step retry;
  - warm-start reset;
  - [x] residual report;
  - abort criteria.

Phase 21 implementation notes:

- Added `atha.runner.DAEExecutionProblem`, `DAEExecutionResult`, and `DAEPoint`.
- The execution object now owns:
  - generic state names and `X0` from `EngineAssembler.initial_vectors()`;
  - algebraic names and `Z0` from `EngineAssembler.port_network_problem()`;
  - schedule evaluation for boundaries, operating targets, timings,
    controllers, and transients;
  - warm-started port-network solves through `NetworkProblem`/`WarmStart`;
  - state, algebraic, normalized residual, command, target, boundary, and
    measurement histories.
- The RHS path now evaluates controllers, transient outputs, and the port
  network together, then returns derivatives for transient blocks and
  controller memory states.
- PI/PID integrals, PID previous-error state, and rate-limiter previous-command
  state are now part of the generic controller-state derivative path.
- `DAEExecutionProblem` now contributes generic physical derivatives for the
  component state paths already declared in the registry:
  - `Pipe.mdot` relaxes toward the solved algebraic pipe flow through the
    YAML `time_constant`;
  - `CombustionChamber.P` and `Preburner.P` integrate finite-volume pressure
    from solved inlet/outlet mass-flow imbalance using `gas_R`, temperature,
    and volume;
  - `Rotor.omega` integrates shaft acceleration from turbine drive power,
    pump load power, friction, and moment of inertia.
- Shaft coupling discovery now follows `engine.yaml` shaft-domain connections
  so a rotor derivative can sum its connected pump loads and turbine drives
  without example-specific topology code.
- The generic measurement dictionary now derives common telemetry outputs from
  solved algebraics and states, including rotor RPM, pump power/load torque,
  turbine drive torque, and nozzle thrust where those values are not already
  explicit algebraic variables.
- `analysis.state_modes` are applied at the generic RHS level. Fixed,
  inactive, and forced-steady states currently zero their derivatives; fixed
  values are also reflected in the state dictionary used by algebraic solves.
- Boundary anchor residuals in `PortNetworkBuilder` now use evaluated
  `boundaries.*` values at each time step. Static boundary anchors continue to
  work when callers solve a port network directly without the DAE execution
  wrapper.
- Added unit coverage proving:
  - time-varying boundary schedules drive port-network algebraic values through
    `DAEExecutionProblem`;
  - PI controller integral states produce the expected derivative;
  - the existing port-network and FFSC acceptance tests remain valid.

Previously blocked deliverables revisited:

- Phase 12 controller state integration is partially unblocked: controller
  state derivatives now exist in the universal execution object. Compatibility
  analyses do not all consume this object yet.
- Phase 12 state mode enforcement is partially unblocked: the generic RHS
  honors state modes, but full ROCETS-style forced steady-state behavior still
  needs physical component derivatives and algebraic state forcing.
- Phase 14 universal histories are partially unblocked: `DAEExecutionResult`
  carries common histories. Existing compatibility runners still return their
  own summaries until migrated.
- Phase 20 time-varying boundary anchoring is unblocked for the port-network
  path.
- Phase 22 rotor and finite-volume dynamics are partially unblocked in the
  universal execution object. These are intentionally conservative derivative
  providers, not yet high-fidelity thermochemical or compressible pipe models.

Remaining blockers:

- Component residual contracts must provide enough equations for a physically
  closed system.
- Fluid property and finite-volume component models must be available before
  FFSC acceptance can move off the reduced-order runner.
- Smaller-step retry, warm-start reset policy, and abort criteria are not fully
  implemented. They need driver-level ownership of failed solver calls and
  phase restarts, which should be added when the first real physical analysis
  migrates to `DAEExecutionProblem`.
- Example 19 still uses the reduced-order FFSC runner. Migrating it now would
  only wrap incomplete pump, pipe, preburner, chamber, turbine, and fluid
  property equations in a nicer loop; the physical residual contracts in
  Phase 22 and Phase 23 must land first.
- Generic DAE execution still needs connected boundary/source nodes for pump
  inlets and external exhausts before the FFSC and GG examples can move from
  reduced runners to `analysis.type: profile`.

### Phase 22: Physical Component Residual Completion

Purpose:

Make the component library strong enough for arbitrary engine networks rather
than only topology diagnostics.

Deliverables:

- [x] Pipe residual contracts:
  - [ ] inertance;
  - [x] friction through quasi-steady pressure-drop closure;
  - [ ] compressibility/storage where configured;
  - optional heat transfer.
- [x] Chamber and preburner finite-volume residuals:
  - [x] mass balance;
  - [x] species or mixture ratio balance through OF closure;
  - [ ] energy balance;
  - [x] pressure/state equation through configurable pressure closure;
  - [ ] outlet state.
- [ ] Merge/junction residuals for multi-inlet chambers and preburners.
- [x] Rotor differential-state contract coupled to pump/turbine torque at the
  `DAEExecutionProblem` derivative level.
- [x] Regen/thermal node contracts with heat-flow residuals.
- [ ] Units and scales on all registry parameters, states, outputs, variables,
  and residuals.

Phase 22 implementation notes:

- Added `PipeMomentumContract`, a quasi-steady incompressible pipe residual
  using either explicit conductance or a Darcy-style friction estimate from
  length, diameter, friction factor, and density.
- Added `FiniteVolumeCombustorContract` for both `CombustionChamber` and
  `Preburner`. It provides first-pass residual closure for:
  - inlet mass accumulation through a component `mdot` algebraic;
  - OF from oxidizer and fuel inlet flows;
  - pressure from the available outlet or inlet port pressure;
  - temperature from `T_adiabatic` or `initial_T`.
- Added `RegenThermalContract` with heat-balance and wall-temperature residuals
  for regen/thermal nodes.
- Updated `ComponentSpec` entries for `Pipe`, `CombustionChamber`,
  `Preburner`, and `RegenChannel` so their algebraic variables, residual
  names, source catalog paths, and contracts are available to the port-network
  assembler.
- Updated `PortNetworkBuilder` so chamber and preburner `mdot` variables are
  not incorrectly linked one-to-one to every inlet and outlet port. Their mass
  closure is now owned by the finite-volume residual contract.
- `EngineAssembler.initial_vectors()` now seeds all registry-declared
  component states, not only states explicitly listed in YAML
  `initial_state`. Defaults are taken from component parameters such as
  `initial_P`, `initial_h`, `initial_mdot`, `mdot_design`,
  `initial_speed_rpm`, and `initial_T_wall`.
- Pipe, chamber/preburner, and rotor state paths are therefore first-class DAE
  `X` entries whenever their component specs declare those states.
- Rotor speed is now treated as a differential state in transient port
  networks rather than as an algebraic torque-balance unknown.
- Added unit coverage for:
  - pipe momentum residual evaluation;
  - chamber finite-volume residual evaluation;
  - example 19 port-network assembly including pipe, chamber, and preburner
    residual names.

Previously blocked deliverables revisited:

- Phase 11 component residual coverage is substantially unblocked for the
  components needed by example 19 topology diagnostics.
- Phase 20 automatic port-network assembly is stronger because pipe,
  chamber/preburner, and regen residuals now contribute equations rather than
  only port variables.
- Phase 21 `DAEExecutionProblem` can now call more physical residual providers,
  but it is still not ready to replace the reduced FFSC runner until map-backed
  turbomachinery and thermodynamic outlet-state equations exist.

Remaining blockers:

- Requires agreement on fluid-property model fidelity: constant-property,
  tabular, CEA/RPA-derived, or external thermodynamics package.
- Pipe inertance and compressibility/storage are not implemented as full
  physical residuals yet. The universal DAE loop now supports first-order
  `Pipe.mdot` state dynamics, but the current pipe contract remains
  quasi-steady and should still be extended with momentum/inertance and
  compressible storage equations.
- Chamber/preburner energy balance and outlet thermodynamic state are not
  implemented. The current contract uses `T_adiabatic`/`initial_T` closure and
  does not compute CEA/RPA-quality combustion products.
- Merge/junction residuals for arbitrary multi-inlet mixing remain open. The
  current finite-volume contract computes OF from named fuel/oxidizer inlet
  ports, but it does not yet produce general enthalpy/species merge equations.
- Rotor differential-state coupling is now present in `DAEExecutionProblem`.
  Remaining rotor work is to improve torque fidelity as pump and turbine
  residuals gain explicit efficiency, enthalpy, and outlet-state equations.
- Units and scales exist on generated `NetworkVariable` and `NetworkResidual`
  objects, but the component registry does not yet enforce unit metadata on
  every YAML parameter.

### Phase 23: Map-Backed Pump And Turbine Physics

Purpose:

Replace reduced affinity-law approximations with reusable map-backed residuals.

Deliverables:

- [x] Load and attach map objects to component map slots during assembly.
- [ ] Pump residuals using corrected speed, corrected flow, pressure rise/head,
  efficiency, torque load, and extrapolation policy.
  - [x] dimensionless `phi`/`psi` pressure rise from attached maps;
  - [x] legacy corrected speed and corrected flow ratios remain as fallback;
  - [x] pump efficiency output from attached `eta`/`efficiency` maps;
  - [x] pump shaft torque residual from solved load power and shaft speed;
  - [x] pump component outlet enthalpy rise from `delta_P / (rho * eta)`;
  - [ ] full extrapolation policy beyond current map clamp behavior.
- [ ] Turbine residuals using pressure ratio, corrected flow, efficiency,
  power/torque drive, and outlet state.
  - [x] pressure ratio and corrected-flow ratio map inputs;
  - [x] efficiency map lookup;
  - [x] power residual scaling from map/design values;
  - [x] turbine shaft torque residual from solved drive power and shaft speed;
  - [ ] turbine outlet thermodynamic state.
- [x] Unit tests with synthetic maps proving outputs change with shaft speed
  and flow.
- [ ] Example 19 acceptance check for pump/turbine map response.

Phase 23 implementation notes:

- `EngineAssembler` and `PortNetworkBuilder` now build runtime
  `PerformanceMap` objects from `loaded.maps` and attach them to component
  model dictionaries by map slot, for example:
  - `lox_pump.map.head_map`;
  - `lox_pump.map.efficiency_map`;
  - `lox_turbine.map.efficiency_map`.
- `PumpHeadContract` now prefers ROCETS-style dimensionless pump maps:
  - flow coefficient `phi = mdot / (rho_design * omega * D^3)`;
  - head coefficient `psi = delta_P / (rho_design * omega^2 * D^2)`.
  It converts `psi` back into pressure rise inside the residual. If a legacy
  map is attached, it still accepts `pressure_rise`, `head`, or `delta_P`, and
  if no map is attached it falls back to the previous affinity-law design
  pressure rise.
- `PumpHeadContract` now returns `pump.efficiency` from attached
  `efficiency_map` outputs named `eta` or `efficiency`, with design efficiency
  as fallback. Port-network assembly filters this output away from strict
  residual checking while preserving it for direct component-contract
  diagnostics.
- `Pump.compute_outputs()` now raises outlet enthalpy using
  `h_out = h_in + delta_P / (rho * eta)` and asks the fluid model for outlet
  temperature when available.
- `ValveFlowContract` now supports passive pressure-driven valves through
  attached `cda_map` or `cd_map` map slots. A `cda_map` can use axes such as
  `inlet.P`, `outlet.P`, `inlet.rho`, `position`, or component-qualified
  paths, and its `CdA` output is used directly as the effective valve area in
  the port residual.
- `TurbinePowerContract` now checks for an attached `efficiency_map` and
  evaluates it with:
  - `pressure_ratio`;
  - `corrected_flow_ratio`;
  - `corrected_flow`.
  It uses the map efficiency with design power/efficiency/pressure-ratio data
  when available, and otherwise falls back to a simple pressure-drop power
  estimate.
- `PortNetworkBuilder` now adds shaft torque residuals for connected pumps and
  turbines:
  - pump shaft torque is linked to load power divided by shaft speed;
  - turbine shaft torque is linked to drive power divided by shaft speed with
    the shaft connection sign convention preserved;
  - rotor shaft-port speed variables are linked to the rotor differential
    state instead of forcing rotor speed through a steady algebraic torque
    balance during transients.
- Fixed `_first_existing()` in `atha.network.ports`, which previously returned
  `None` after the first missing candidate path and could skip valid fallback
  inlet/outlet pressure paths.
- Converted example 19 pump maps from speed-ratio/flow-ratio pressure-rise
  tables to single-axis `phi` maps with `psi` and `eta` outputs.
- Converted example 20 pump maps from analytic constant maps to CSV-backed
  one-axis `phi` tables with constant `psi`/`eta` values matching the previous
  analytic outputs.
- Example 20's reduced GG runner now loads those runtime pump maps directly and
  computes pump pressure rise from absolute shaft speed using
  `delta_P = rho * psi(phi) * omega^2 * D^2`, rather than applying a
  normalized speed-ratio multiplier to `dP_design`.
- Example 20 no longer treats `analysis.initial_conditions.shaft.rpm` as the
  pump design speed. The reduced runner uses pump `speed_design` (or explicit
  `analysis.design.shaft_speed_design_rpm`) as the fixed reference for reduced
  flow capacity and shaft friction.
- Added `rho_design`, pump diameter, and `head_map`/`efficiency_map` bindings
  to examples 19 and 20 so the revised maps load through the same runtime map
  infrastructure.
- Examples 19 and 20 now export pump efficiency and pump inlet/outlet enthalpy
  telemetry so the map/enthalpy behavior can be checked from CSV output.
- Example 20 additionally exports pump `delta_P`, `phi`, and `psi` telemetry so
  off-design shaft-speed behavior is visible in the CSV and pump-head plot.
- The reduced FFSC algebraic solve keeps pump efficiency and enthalpy as
  derived telemetry outputs instead of adding them as independent residual
  unknowns.
- Added unit tests proving:
  - pump residuals change with map-driven speed/flow ratios;
  - turbine residuals change with map-driven corrected flow and efficiency;
  - valve residuals can use a pressure-dependent passive `CdA` map.
- Re-ran examples 19 and 20 with the revised pump maps:
  - example 19 wrote `outputs/codex-example19/ffsc_dae_acceptance.csv` and
    passed its acceptance report;
  - example 20 wrote `outputs/codex-example20/gg_single_shaft_methalox.csv`;
  - example 20 now loads `lox_pump_affinity.csv` and
    `methane_pump_affinity.csv` instead of using constant-source map YAML;
  - with the current `24000 rpm` initial shaft speed, example 20 starts at
    `6.75 MPa` LOX pump pressure rise and `7.04 MPa` methane pump pressure
    rise, below the `32000 rpm` design-speed values produced by the same
    `psi` maps;
  - both examples now run through `30.0 s`, with shutdown beginning at `25.0 s`;
  - example 19 crossover commands and example 20 gas-generator commands are
    zero at the first telemetry sample at or after `25.0 s`;
  - both runs exported `LOX_PUMP_EFFICIENCY = 0.74` and
    `METHANE_PUMP_EFFICIENCY = 0.69`;
  - minimum outlet-minus-inlet pump enthalpy rise was positive in both examples
    (`15.4 kJ/kg` LOX and `31.2 kJ/kg` methane for example 19, `6.4 kJ/kg`
    LOX and `19.5 kJ/kg` methane for example 20).

Previously blocked deliverables revisited:

- Phase 11 pump/turbine map-value response is now unblocked at the residual
  contract level.
- Phase 20/21 port-network diagnostics now have access to actual loaded map
  objects when assembling component residual contracts.
- Example 19's source-catalog/port-network assembly path now loads its pump and
  turbine map files during validation.

Remaining blockers:

- Requires final fluid-property conventions for corrected flow and corrected
  speed in non-pump turbomachinery maps.
- Pump efficiency is exposed as a component output, not as an independent
  algebraic residual variable. A future high-fidelity pump model may promote
  efficiency, torque, and outlet state into a coupled residual block.
- Pump and turbine shaft torque residuals now exist, but turbine outlet
  thermodynamic state and full energy closure remain open.
- Turbine outlet thermodynamic state is still blocked by the fluid-property and
  energy-balance model decisions from Phase 22.
- Example 19 still validates shaft response and reduced-model pump/turbine
  behavior, but it does not yet include a full port-variable acceptance check
  that proves loaded map interpolation changes the solved FFSC transient.

### Phase 24: Generic Trim, Profile, And Linearization Modes

Purpose:

Move ROCETS-style run modes out of example-specific handlers.

Deliverables:

- [x] `analysis.type: steady` performs port-network trim/diagnostics with named
  residuals.
- [x] `analysis.type: profile` runs multi-phase transient profiles using the
  universal DAE loop.
- [x] `analysis.type: linearization` linearizes any generic DAE problem and
  exports state-space matrices.
- [ ] User-defined algebraic balances in YAML, including targets and varied
  variables.
- [ ] ROCETS-style state forcing to steady-state during reduced-order studies.

Phase 24 implementation notes:

- Added `atha.analysis.generic_modes`, including:
  - `run_generic_steady`;
  - `run_generic_profile`;
  - `run_generic_linearization`;
  - `GenericDAESummary`.
- Registered generic `analysis.type` handlers:
  - `steady`;
  - `profile`;
  - `linearization`.
- `steady` builds a `DAEExecutionProblem`, evaluates the configured port
  network at the initial time, and writes a JSON diagnostic artifact containing
  states, algebraics, residuals, commands, targets, boundaries, and
  measurements.
- `profile` integrates through `DAEExecutionProblem`, exports generic
  diagnostics JSON, and when `telemetry.yaml` is provided, writes telemetry CSV,
  plot, HDF5, residual diagnostics, and manifest artifacts through
  `OutputProcessor`.
- `linearization` finite-differences the generic DAE RHS and exports state-space
  matrices with `A`, `B`, `C`, `D`, labels, operating point, and perturbation
  sizes.
- `NetworkProblem.solve()` now handles empty algebraic networks cleanly, which
  allows transient-only linearization cases such as a commanded valve response.
- Added unit coverage proving:
  - `steady` runs through the registry and writes diagnostics;
  - `profile` exports telemetry from the universal DAE loop;
  - `linearization` exports a state-space JSON artifact.

Previously blocked deliverables revisited:

- Phase 7 and Phase 12 profile and linearization registry modes are now
  unblocked.
- Phase 14 generic output is further unblocked for analyses that run through
  `profile`, because they now use the shared output processor.
- Phase 21 universal DAE histories now have a user-facing analysis mode through
  `analysis.type: profile`.

Remaining blockers:

- User-defined algebraic balances are not implemented. They need a safe
  expression/balance schema tied into `NetworkProblem` residual construction.
- The current `steady` mode is a port-network trim/diagnostic solve, not a full
  target-balance optimizer that varies selected commands to hit operating
  targets such as Pc, thrust, mdot, or OF.
- ROCETS-style forced steady-state is represented in `state_modes` and
  derivative zeroing, but full reduced-order state forcing needs component
  derivative providers and explicit steady-state residual substitution.
- Generic linearization works for the DAE execution object, but high-fidelity
  FFSC linearization remains blocked until example 19 migrates off the
  reduced-order runner and into the full port-variable DAE path.

### Phase 25: Generic Sweep And Monte Carlo Perturbations

Purpose:

Make sweeps and Monte Carlo independent of gas-generator-specific patch logic.

Deliverables:

- [x] Normalize `analysis.perturbations` using dotted paths and
  `apply_path_overrides()`.
- [x] Support structured sweeps over component parameters, map scale factors,
  boundary conditions, controller gains, and initial conditions.
- [x] Support random distributions over any YAML path.
- [x] Export varied parameter tables, ensemble statistics, and sensitivity
  metrics through the output processor.

Phase 25 implementation notes:

- Added generic registered analysis types:
  - `analysis.type: sweep`;
  - `analysis.type: monte_carlo`.
- Both modes wrap a `base_type` analysis such as `profile`, `steady`, or
  `linearization`, apply case-specific YAML overrides to a copied
  `LoadedAnalysisConfig`, build a fresh execution plan, and run through the
  same analysis registry.
- Added support for `analysis.perturbations.sweep` with either explicit
  `values`, a scalar `value`, or `start`/`stop`/`points`.
- Added support for `analysis.perturbations.monte_carlo` with seeded
  distributions:
  - `uniform`;
  - `normal`;
  - `lognormal`;
  - `choice`.
- Extended `apply_path_overrides()` so dotted YAML keys such as
  `boundary_conditions.conditions.lox_tank.outlet.P.value` can be overridden
  without requiring users to rename physical telemetry or boundary paths.
- Added generic metric extraction from child telemetry CSV outputs using
  configured metrics with `final`, `initial`, `mean`, `min`, and `max`
  reducers. When no metric list is supplied, numeric final telemetry channels
  are exported as metrics.
- Added ensemble artifacts:
  - case table CSV containing varied parameters and metrics;
  - statistics JSON with min, max, mean, and standard deviation;
  - sensitivity CSV with Pearson correlation for numeric varied parameters and
    numeric metrics;
  - sweep manifest JSON.
- Added tests for:
  - registry discovery of `sweep` and `monte_carlo`;
  - dotted YAML-key overrides;
  - structured sweep execution through `profile`;
  - seeded Monte Carlo execution through `profile`;
  - example 19 acceptance staying functional.

Previously blocked deliverables revisited:

- Phase 12 sweep/Monte Carlo modes are now unblocked for analyses that can run
  through the public registry and expose metrics through telemetry CSV output.
- Phase 16 arbitrary perturbation support is further unblocked because the
  override primitive now handles dataclass paths, dict/list paths, and YAML
  keys that themselves contain dots.
- Phase 19 result artifacts are extended with generic statistics and
  sensitivity outputs.

Remaining blockers:

- Sweep/Monte Carlo metrics are stable for telemetry-backed analyses. Metrics
  from analyses without telemetry still require each summary object to expose a
  consistent metrics payload or residual diagnostics.
- No parallel execution is implemented yet. This is a deliberate follow-up
  because nested analyses may write artifacts and use non-thread-safe external
  thermochemistry backends.
- Global optimizer/trim sweeps are not implemented. Phase 25 varies YAML
  values and evaluates cases; it does not yet solve for a parameter value that
  satisfies a target balance.

### Phase 26: Validation And Golden Regression Suite

Purpose:

Prove solver behavior against known systems before relying on FFSC acceptance
results.

Deliverables:

- [x] Golden outputs for examples 15-18.
- [x] Shutdown transient validation case.
- [x] Component-level residual closure tests for every residual provider.
- [x] Subsystem validation for valve-pipe-volume, pump-shaft-turbine, chamber
  and nozzle, and preburner-turbine chains.
- [ ] Example 19 full port-variable FFSC acceptance gate.
- [x] Test-data comparison workflow for imported CSV/HDF5 telemetry.

Phase 26 implementation notes:

- Added `atha.validation.regression` with:
  - `MetricWindow`;
  - `RegressionCheck`;
  - `RegressionReport`;
  - `build_regression_report()`;
  - `build_regression_report_from_file()`;
  - `write_regression_report_json()`;
  - built-in regression windows for examples 15-18.
- Added `atha.validation.residual_closure` with:
  - `evaluate_component_residual_closure()`;
  - `assert_component_residual_closure()`.
- Added telemetry import/compare workflow in `atha.output.comparison`:
  - `load_time_series_csv()`;
  - `load_time_series_hdf5()`;
  - `load_time_series()`;
  - `compare_time_series_files()`.
- Telemetry export now writes `NaN` for sampled diagnostic sources that are
  intentionally absent in an inactive controller phase, instead of aborting the
  entire output write.
- Added integration regression tests for examples 15-18. These cover
  valve-volume, valve-pipe-injector-chamber-nozzle, and pressure-fed TCA
  behavior. They are tolerance
  windows over final/max behavioral metrics rather than byte-for-byte golden
  CSV files, because examples 15-18 are still compatibility analyses and their
  output sampling/plot artifacts should not be frozen as permanent file
  baselines yet.
- Added residual closure coverage for all currently registered component
  residual providers:
  - `Valve`;
  - `Nozzle`;
  - `MassFlowInjector`;
  - `FlowSplitter`;
  - `Pipe`;
  - `CombustionChamber`;
  - `RegenChannel`;
  - `Pump`;
  - `Turbine`;
  - `Rotor`.
- Added focused unit coverage for:
  - PID derivative sampling from the previous controller sample and division by
    controller `dt`;
  - `previous_error` no longer being registered as an ODE state;
  - `active_phases` suppressing controller output outside the named phase;
  - pump `phi`/`psi` map residual closure;
  - pump efficiency output and outlet enthalpy rise.
- Reverified example 19 acceptance after the Phase 25 controller-sign fix.
- Added sampled controller evaluation with `controllers.evaluation`:
  - `frequency_hz`;
  - `period_s`.
- Example 19 now runs controller updates at `20 Hz` and includes main
  propellant and crossover valve shutdown at `25 s`. The reduced-order acceptance report is
  evaluated over the powered portion before shutdown so endpoint thrust checks
  do not conflict with the requested shutdown transient.
- Added example 20, a methalox single-shaft gas-generator cycle with pump,
  pipe, splitter, main valve, injector, chamber, nozzle, generator branch,
  turbine, and ambient exhaust paths. Example 20 uses the same sampled
  controller concept and closes both main propellant and gas-generator valve
  setpoints at the `25 s` shutdown phase boundary.
- Added an integration gate for example 20 that checks the engine reaches a
  powered thrust level and then decays after shutdown.
- Reverified example 19 and 20 after the Phase 23 pump-map and Phase 13
  phase/PID updates:
  - focused PID/phase/pump-map unit tests: `8 passed`;
  - example 19 CLI run: acceptance `PASS`;
  - example 20 CLI run: completed and exported CSV/HDF5/manifest/plot artifacts;
  - example 20 final thrust after the aligned shutdown timing was
    `20.7 kN`, below half of the `149.9 kN` peak;
  - example 20 produced varying CLC PID derivative telemetry:
    LOX generator derivative ranged from `-54.86` to `52.75`, and methane
    generator derivative ranged from `-1.34` to `1.66`.
- Reverified example 20 PID telemetry after the sampled-integral fix:
  - `raw_command = bias + proportional_term + integral_term + derivative_term`
    reconstructs to floating-point precision for both PID loops;
  - LOX loop integral is nonzero but held nearly constant by anti-windup while
    the command is saturated;
  - methane loop integral evolves over CLC from about `-0.023` to `4.188`.

Previously blocked deliverables revisited:

- Phase 14/19 telemetry artifact validation is further unblocked because CSV
  and HDF5 can now be loaded through one comparison API.
- Phase 17 validation workflow documentation is supported by concrete report
  formats for acceptance, regression, residual diagnostics, and telemetry
  comparison.
- Phase 22 residual provider validation is now partly unblocked by reusable
  residual closure helpers and tests for every registered provider.

Remaining blockers:

- Shutdown transient validation is implemented as an example-level regression
  gate for example 20. A higher-fidelity shutdown benchmark remains future work
  once blowdown physics and valve close criteria are agreed.
- Subsystem validation is now covered by examples 15-18 and the reduced-order
  gas-generator example 20. A universal-port DAE pump-shaft-turbine benchmark
  remains desirable after the full port-variable solver owns turbomachinery
  torque closure.
- Example 19 still uses the reduced-order FFSC DAE acceptance runner. The
  "full port-variable FFSC acceptance gate" remains blocked until the generic
  port DAE owns all FFSC states, all algebraic port variables, and high-fidelity
  pump/turbine/preburner/chamber closures.
- Byte-for-byte golden CSV/HDF5 baselines for examples 15-18 remain deferred.
  Current validation uses metric windows, which is the appropriate level while
  those examples are still being migrated toward the universal DAE execution
  path.

### Phase 27: Documentation And User Libraries

Purpose:

Make the tool usable without Codex rewriting Python for every new model.

Deliverables:

- [x] YAML include/library mechanism for component templates, maps,
  controllers, telemetry groups, and subsystem fragments.
- [x] Full schema reference with units and examples.
- [x] Contributor guide for residual providers, map-backed components, and
  validation tests.
- [x] Worked examples for pressure-fed, gas-generator, staged-combustion,
  FFSC, sweep, Monte Carlo, and linearization analyses.

Phase 27 implementation notes:

- Added pre-schema YAML include expansion in `atha.config.loader`.
  Supported keys:
  - `include`;
  - `$include`.
  Includes may be a path string or ordered list of path strings and are
  resolved relative to the YAML file that declares them.
- Include merge policy:
  - mappings are deep-merged;
  - local scalar values override included scalar values;
  - lists append, enabling reusable `connections`, telemetry `channels`,
    timing `events`, and plot lists;
  - include cycles raise `ConfigError` with the include chain.
- Added unit coverage for:
  - component-fragment reuse in `engine.yaml`;
  - telemetry channel-group reuse in `telemetry.yaml`;
  - include cycle detection.
- Added documentation:
  - `docs/yaml_libraries.md`;
  - `docs/schema_reference.md`;
  - `docs/contributor_guide.md`;
  - `docs/worked_examples.md`.

Previously blocked deliverables revisited:

- Phase 8 YAML library support is now unblocked at the file-composition layer.
  A config folder can reuse valve/component fragments and telemetry channel
  groups from another YAML file.
- Phase 14/19 diagnostic documentation is now reflected in the schema and
  worked-example docs, including CSV/HDF5, plots, residual diagnostics, and
  acceptance artifacts.
- Phase 17 validation workflow documentation is now covered by the contributor
  guide and worked-example notes.

Prior incomplete deliverable sweep:

- Phase 8 reusable libraries are implemented for file-level YAML composition.
  Higher-level named templates remain a separate feature because they need a
  schema for parameters, defaults, substitution, and validation of generated
  fragments before merge.
- Phase 9 full `extract_engine_model()` replacement remains blocked. The
  assembler can build source catalogs, initial vectors, and port diagnostics,
  but compatibility runners still need temporary model dictionaries until every
  component has a registry constructor and all examples consume the same
  assembled engine object.
- Phase 11 multi-inlet merge residuals remain blocked by the thermodynamic
  closure work. Implementing them safely requires mixture state variables,
  enthalpy/species balance policy, and outlet-state definitions for chamber and
  preburner ports.
- Phase 12 and Phase 20 full arbitrary port-variable FFSC transient execution
  remain blocked by component closure fidelity, not by topology assembly. The
  port network can generate and diagnose unknowns/residuals; the next unblocker
  is replacing reduced pump/turbine/chamber/preburner equations with
  port-coupled residual providers that also produce state derivatives.
- Phase 17 example 19 full port-variable acceptance remains blocked for the
  same reason. The reduced DAE acceptance gate should stay in place until the
  universal port solver can match its powered-operation metrics and shutdown
  behavior.
- Phase 19 generic aliases are partly unblocked through `steady`, `profile`,
  `linearization`, `sweep`, and `monte_carlo`. Remaining alias work depends on
  migrating legacy handlers to context-aware execution after the universal DAE
  owns the shared state, algebraic, telemetry, and artifact histories.
- Phase 24 user-defined balances remain blocked by expression safety. The
  unblocker is a typed balance schema that binds only known source-catalog paths
  and compiles to named `NetworkProblem` residuals without evaluating arbitrary
  Python.
- Phase 26 full FFSC golden acceptance remains blocked until example 19 runs
  through the full port-variable solver. Existing metric-window regressions for
  examples 15-18 and example 20 should remain the validation gate while those
  examples still include compatibility behavior.

Remaining blockers:

- Named, parameterized template catalogs are not implemented. The include
  mechanism provides merge semantics, but parameterized subsystem generation
  needs a dedicated template schema and substitution policy.
- User-defined algebraic balances remain blocked on a safe expression/balance
  schema connected to `NetworkProblem` residual construction.
- Full FFSC worked-example documentation is limited to the current reduced DAE
  acceptance workflow until the full port-variable FFSC solver replaces the
  reduced runner.

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
