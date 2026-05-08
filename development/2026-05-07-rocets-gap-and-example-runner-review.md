# ROCETS Gap And Example Runner Review

Date: 2026-05-07

## Scope

This review compares the current ATHA overhaul branch against the NASA ROCETS
final report in `resources/19910011919.pdf`, with emphasis on:

- whether the current YAML/runtime architecture matches the ROCETS intent;
- where example `run.py` files have drifted into framework code;
- current transient behavior, especially the TCA PMS examples.

## ROCETS Architecture Signals

The ROCETS report describes a simulation system with these major ideas:

- **Reusable component library.** The report summary describes engine
  simulations generated from stored component modules through high-level input.
  The delivered library was reported as 24 component modules, 57 submodules and
  maps, and 33 system routines/utilities.
- **Separate processors.** The system is organized around Configuration, Run,
  Execution, and Output processors. The Run Processor reads experiment input,
  initializes inputs, sets flags, accepts schedules, defines additional
  balances, and selects run modes. The Execution Processor owns looping,
  balancing, transient integration, and linearization. The Output Processor owns
  print/plot channel selection.
- **Three run modes.** ROCETS explicitly supports steady-state trim balance,
  transient operation, and linearization.
- **Coupled nonlinear transient solve.** In transient mode, ROCETS integrates
  dynamic states with predictor/corrector methods and closes corrector
  equations and algebraic balances simultaneously.
- **State activation modes.** ROCETS allows states to be active, inactive, or
  forced to steady state. This matters for transient model conditioning and for
  turning selected dynamics on/off during a run.
- **Schedule-driven experiments.** The Run Processor supports schedule/curve
  inputs and run-specific integration options.
- **Closed-loop control integration.** The TTBE verification included an
  interfaced NASA FORTRAN control model operated in closed-loop transient mode.

## Current ATHA Alignment

ATHA now aligns with ROCETS in several important structural ways:

- Engine topology is separate from analysis execution through Engine YAML.
- Boundary conditions, operating conditions, timings, controllers, maps, and
  telemetry are separate configuration concepts.
- Performance maps can be supplied as independent files and bound into
  component slots.
- Steady trim and transient execution exist as separate solver modes.
- Examples can now consume target profiles as external JSON/CSV data, which
  matches the ROCETS schedule/curve-input concept better than hard-coded Python.
- Examples 13 and 14 now use source-level shared runner code under
  `atha.examples`, reducing local `run.py` files to thin entrypoints.

## Technical Gaps

### 1. Execution Processor Is Still Split Across Examples

ROCETS centralizes run execution in the Execution Processor. ATHA still has
too much execution knowledge scattered through examples 04, 09, and 10:

- cycle-specific engine builders still live in `run.py`;
- Monte Carlo and sweep orchestration still live in example scripts;
- output writing and plotting are partially repeated;
- steady initialization policies are chosen per script.

The new `atha.examples.tca` runner fixes this for examples 13 and 14, but the
larger GG/FFSC examples still need equivalent source-level runners.

### 2. Configuration Processor Is Partial

ROCETS generated a main program from structured configuration and automatically
scanned required inputs and algebraic balances. ATHA loads YAML and validates
map/transient references, but does not yet:

- instantiate arbitrary component classes from YAML type names;
- infer required boundary inputs from component ports;
- detect algebraic loops in the assembled graph;
- generate or assemble a global square algebraic system from connections;
- validate telemetry channel sources against model outputs.

### 3. Transient Solver Coupling Is Not ROCETS-Equivalent

ROCETS closes transient corrector equations and algebraic balances
simultaneously. ATHA currently uses SciPy `solve_ivp` with RHS evaluation and
limited component algebraic solving. Connection residuals are registered but
not solved as a full global algebraic system.

This is the biggest transient-fidelity gap. It shows up in example 14: valve
timing is modeled as a command gate on mass-flow injectors rather than as a
pressure-network flow solution through valve, injector, chamber, and nozzle
algebraic coupling.

### 4. State Activation And Transient Component Models Are Missing

ROCETS supports state ON/OFF/STEADY-STATE selections and lets users remove
selected dynamic effects during a run. ATHA does not yet expose:

- per-state activation from YAML;
- forced steady-state state handling during transient integration;
- reusable transient blocks such as first-order lag and rate-limit behavior
  bound directly to component parameters;
- event-triggered activation/deactivation.

The current valve example uses schedule ramping, not a true component transient
state.

### 5. Control System Layer Is Prototype-Level

ATHA has YAML controller definitions and simple controller types:

- `null`;
- `of_mass_flow_split`;
- `gain_product`.

ROCETS demonstrated closed-loop operation with an external NASA control model.
ATHA still needs a real controller processor:

- PID blocks;
- filters and lags;
- limits, deadbands, and rate limits;
- controller state histories;
- sensor/actuator aliases;
- external controller/plugin interface;
- control targets solved against operating conditions.

### 6. Output Processor Is Minimal

Telemetry YAML exists, but current examples mostly write fixed CSV columns from
runner code. A ROCETS-like Output Processor should:

- read telemetry channel definitions;
- sample at configured rates;
- export CSV/HDF5 consistently;
- apply aliases and units;
- handle derived outputs and missing channels clearly;
- keep plotting separate from run execution.

### 7. Performance Validation Is Not Yet ROCETS-Level

ROCETS used the TTBE model as a verification vehicle, including steady main
stage, throttle transients, start, shutdown, and closed-loop control operation.
ATHA has tests and simplified examples, but no ROCETS/TTBE transient benchmark
comparison is currently encoded.

## Current Example Performance

The current generated outputs give these metrics:

| Example | Time span | Pc range | Thrust range | Max dPc/dt | Max dF/dt |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 TCA PMS runbox | 0.0 to 7.25 s | 7.342-11.013 MPa | 13.976-20.964 kN | 2.10 MPa/s | 4.00 kN/s |
| 14 TCA valve timing | -2.0 to 7.25 s | 0.097-11.013 MPa | 0.178-20.964 kN | 13.81 MPa/s | 26.29 kN/s |

Interpretation:

- Example 13 is smooth because it starts from a steady trim at the first PMS
  target and only follows the target profile perimeter.
- Example 14 includes a closed-valve dwell and a valve-opening ramp, but it
  still has a steep early transient because the simplified chamber/nozzle model
  is not solving a coupled valve/injector/nozzle pressure network. The current
  command-gate approximation is useful for testing the YAML/timing/controller
  shape, not for ROCETS-grade startup transient fidelity.
- Metrics above are from telemetry exports sampled at the YAML
  `sample_rate_hz`, not raw adaptive solver internal steps.

## Refactor Completed In This Pass

The TCA examples were reduced substantially:

- `examples/13_tca_pms_runbox/run.py`: 33 lines
- `examples/14_tca_pms_valve_timing/run.py`: 35 lines
- shared TCA execution now lives in `atha/examples/tca.py`
- shared utilities now live in `atha/examples/common.py`

This is closer to the ROCETS separation:

- `run.py` is now a thin run entrypoint;
- YAML owns run configuration;
- source-level runner code owns execution mechanics;
- output generation is no longer copied between examples 13 and 14.

Additional fixes started after this review:

- Controller block evaluation moved to `atha/config/controllers.py`.
- TCA CSV exports now use `telemetry.yaml` aliases and units through
  `atha/output/telemetry.py`.
- TCA telemetry export now respects `sample_rate_hz` using interpolated state
  samples before output evaluation.
- Example 13 and 14 telemetry now define LOX/fuel mass-flow channels; example
  14 also defines valve-position channels.
- TCA component construction now uses `atha/components/factory.py` for the
  supported YAML component types: `Valve`, `MassFlowInjector`,
  `CombustionChamber`, and `Nozzle`.

These changes are early slices of the ROCETS Run, Output, and Configuration
Processor responsibilities. They are not yet the complete processors.

## Recommended Implementation Plan

1. Add a generic component factory.
   Instantiate engine components from YAML type names and registered builders,
   instead of writing a custom `build_engine()` per example.
   Status: started for TCA component types. Remaining work is pump, turbine,
   rotor, preburner, gas generator, regen, orifice, maps, and cycle-specific
   backend injection.

2. Promote a real Analysis Runner.
   Add `atha.analysis.runner` or `atha.execution` to own steady trim,
   transient integration, sweeps, Monte Carlo, timing evaluation, controller
   evaluation, telemetry sampling, and export.
   Status: started for TCA profiles under `atha.examples.tca`. Remaining work
   is a cycle-agnostic runner and moving examples 04, 09, and 10.

3. Implement telemetry export from YAML.
   Replace hard-coded CSV columns in examples with telemetry YAML channel
   evaluation and aliases.
   Status: started for TCA examples with aliases, units, and sample-rate CSV
   export. Remaining work is HDF5 export, robust channel validation, and
   adoption by all examples.

4. Implement transient parameter blocks.
   Bind `transients/*.yaml` to component commands/parameters for first-order
   lag, rate limits, saturation, and event activation.
   Status: not started.

5. Assemble global algebraic connection solve.
   Use the existing DAE foundation to solve connection residuals and component
   algebraic residuals together. This is the key step toward ROCETS-like
   transient behavior.
   Status: not started. This is the largest numerical change.

6. Upgrade valve/injector/nozzle startup physics.
   Replace command-gated mass-flow injectors with pressure-driven valve and
   injector flow models in example 14 once the global algebraic solve is
   available.
   Status: blocked on the global algebraic connection solve for ROCETS-grade
   fidelity.

7. Add ROCETS/TTBE validation cases.
   Encode steady and transient reference cases from the ROCETS report so ATHA
   performance can be compared quantitatively, not only structurally.
   Status: not started.

8. Refactor examples 04, 09, and 10.
   Move repeated cycle builders, Monte Carlo/sweep output, and plot/export code
   into source modules. Leave `run.py` files as thin entrypoints like examples
   13 and 14.
   Status: not started in this pass.
