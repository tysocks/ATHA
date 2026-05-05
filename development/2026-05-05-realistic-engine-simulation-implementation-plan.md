# Realistic Rocket Engine Simulation Implementation Plan

Date: 2026-05-05

## Purpose

ATHA is intended to become a Python library similar in spirit to ROCETS: a
component-based simulation environment for modelling liquid rocket engine
cycles, steady operating points, transient behavior, step changes, and
closed-loop controls in a way that is as close to real engine behavior as is
practical for cycle-level engineering analysis.

This document is written as an implementation brief for an AI agent. It
describes the target capability, the major missing pieces in the current
project, the architecture that should be implemented, and the order in which
work should be done. The intent is to reduce ambiguity and prevent agents from
adding isolated features that do not fit the required physical and numerical
model.

## Current Project Summary

The repository already contains the foundation of the library:

- `atha/core`: engine graph, components, ports, and `Engine.compile()`.
- `atha/solver`: steady-state and transient solvers.
- `atha/components`: volumes, chambers, pumps, turbines, valves, nozzles,
  pipes, rotors, regen channels, splitters, and thermal nodes.
- `atha/thermo`: ideal gas, CoolProp, and Cantera backends.
- `atha/maps`: generic performance map interpolation.
- `atha/profiles`: multi-phase profile execution with scheduled commands and
  safety limits.
- `atha/monte_carlo`: sampling, execution, statistics, and sensitivity tools.
- `atha/jannaf`: simplified JANNAF performance calculations.
- `atha/validation`: TTBE-style validation support.

The current implementation is useful as a scaffold, but it is not yet a true
high-fidelity engine-system solver. The largest gap is that the documented
architecture describes a DAE system with algebraic unknowns eliminated by an
inner Newton solve, while the implemented transient solver mostly performs two
context propagation passes over dynamic states. This means the solver can work
for simple feed-forward examples, but it is not yet reliable for realistic
engine cycles with algebraic loops, coupled shaft power balance, injector and
nozzle choking, staged-combustion recirculation, or closed-loop control.

## Target Capability

The completed library should support the following use cases.

### 1. Engine Cycles And Components

Users must be able to construct engine cycles by connecting reusable
components, not by selecting hard-coded cycle classes. The same component and
solver infrastructure should support at least:

- Pressure-fed engines.
- Gas generator cycles.
- Expander cycles.
- Oxidizer-rich staged combustion.
- Fuel-rich staged combustion.
- Full-flow staged combustion.
- Multi-shaft or geared turbomachinery arrangements.
- Bleeds, bypasses, recirculation lines, chilldown paths, and purge paths.

Cycle behavior must emerge from equations and topology. Component insertion
order must not change the solved physical result.

### 2. Steady-State Operating Points

The library must solve steady operating points for a given set of boundary
conditions, actuator positions, shaft constraints, and targets. It should be
able to find trim points such as:

- Chamber pressure target.
- Thrust target.
- Mixture ratio target.
- Pump outlet pressure target.
- Turbine power balance.
- Shaft speed target.
- Valve position required for a flow split.
- Preburner temperature or mixture ratio target.

Steady-state solves must include algebraic variables, not only dynamic state
derivatives. A steady-state residual should include:

- `dX/dt = 0` for dynamic states.
- `Rz = 0` for algebraic component equations.
- Connection constraints.
- Trim target residuals.
- Optional controller steady residuals when controllers are enabled.

### 3. Transient Behavior

Transient simulations must integrate dynamic states through time while solving
instantaneous algebraic constraints at every RHS evaluation. Required transient
phenomena include:

- Tank blowdown or feed pressure decay.
- Volume pressurization and depressurization.
- Valve opening and closing.
- Pump spin-up and spin-down through shaft torque balance.
- Turbine startup and shutdown.
- Chamber ignition, mainstage, throttling, and shutdown.
- Nozzle throat choking and ambient pressure changes.
- Regenerative wall heating and coolant temperature rise.
- Controller action and actuator lag.

### 4. Step Changes And Events

Users must be able to model real step changes and event-driven transitions:

- Valve command steps.
- Pump speed command steps.
- Turbine bypass command steps.
- Ignition event.
- Nozzle ambient pressure step or ramp.
- Tank isolation or line opening.
- Abort trigger.
- Phase transition when a measured condition crosses a threshold.

Discontinuities should be handled explicitly by splitting integration segments
at event times or by using event functions. Do not hide discontinuities inside
arbitrary schedule functions without resetting solver step history.

### 5. Closed-Loop Control

The library must support control schemes that read simulated sensor values,
compare them to targets, and change actuator commands. At minimum:

- PID controllers.
- Setpoint schedules.
- Sensor definitions.
- Signal filtering.
- Actuator limits.
- Rate limits.
- Anti-windup.
- Manual override.
- Bumpless transfer between manual and automatic modes.
- Controller enable/disable windows inside test profiles.

Controllers should command physical actuator targets, not directly overwrite
physical states unless a special ideal-control mode is explicitly requested.

## High-Level Architecture To Implement

Implement ATHA as an index-1 DAE engine simulator:

```text
User model
    |
    v
Engine graph
    |
    v
Compiled layout
    |
    +-- dynamic state vector X
    +-- algebraic unknown vector Z
    +-- command/input vector U
    +-- variable registry and residual registry
    |
    v
Evaluation
    |
    +-- dXdt = f(t, X, Z, U)
    +-- Rz  = g(t, X, Z, U)
    +-- telemetry outputs
    |
    v
Solvers
    |
    +-- steady trim: solve [dXdt, Rz, targets] = 0
    +-- transient: solve Rz = 0 inside RHS, integrate dXdt
```

## Critical Missing Piece: First-Class Algebraic Variables

The current code has `n_algebraic` and `alg_offsets`, but components generally
do not define algebraic unknowns or residuals. This must change.

### Variable Types

Introduce a variable registry during `Engine.compile()` with four categories.

#### Dynamic States `X`

Dynamic states are integrated through time:

- Volume pressure and enthalpy: `component.P`, `component.h`.
- Pipe inertial mass flow: `component.mdot`.
- Rotor speed: `component.omega`.
- Wall or metal node temperature: `component.T_wall`.
- Valve actual position: `component.position`.
- Actuator internal states if needed.
- Controller integrator state and filtered derivative state.
- Tank mass, ullage pressure, or pressurant state when implemented.

#### Algebraic Unknowns `Z`

Algebraic unknowns are solved at a fixed time and state:

- Port pressure, enthalpy, temperature, density, and mass flow.
- Shaft port speed and torque.
- Thermal port heat flow and wall/interface temperature.
- Pump outlet pressure, head, efficiency, torque, and outlet enthalpy.
- Turbine outlet pressure, corrected flow, efficiency, torque, and outlet
  enthalpy.
- Nozzle throat mass flow and pressure relation.
- Valve or injector mass flow from pressure drop and area.
- Flow split variables when split fraction is not fixed.
- Boundary unknowns used for trim.

#### Commands And Inputs `U`

Commands and inputs are values supplied by profiles, controllers, or external
boundary conditions:

- Tank pressures and temperatures.
- Ambient pressure.
- Valve command position.
- Pump speed command.
- Controller setpoints.
- Ignition or heat release command.
- External load or imposed shaft speed for test rigs.

#### Parameters

Parameters are fixed during a solve unless a profile explicitly changes them:

- Areas, lengths, volumes, inertias, roughness, material properties.
- Map objects and map scaling factors.
- Efficiencies, leakage coefficients, deadbands, limits.

### Variable Metadata

Every registered variable must have metadata:

- Name: globally unique string such as `fuel_pump.outlet.P`.
- Kind: state, algebraic, command, parameter, output.
- Units: SI unit string.
- Scale: numerical scale for residual normalization.
- Bounds: optional lower and upper bounds.
- Description.
- Owner component.

These metadata are required for diagnostics and robust nonlinear solving.

## Global Residual Assembly

Replace the current context propagation model with an explicit residual
assembly model.

### Required API

Add an evaluation API similar to:

```python
class EngineLayout:
    def evaluate(self, t, X, Z, U) -> EvaluationResult:
        ...

@dataclass
class EvaluationResult:
    dXdt: np.ndarray
    Rz: np.ndarray
    outputs: dict[str, float]
    residual_names: list[str]
    output_names: list[str]
```

Components should contribute equations through a contract similar to:

```python
class BaseComponent:
    def register_variables(self, registry): ...
    def evaluate(self, t, view): ...
```

The `view` object should allow components to read and write named variables
without knowing global indices. The compiled layout should translate these
named reads and writes into array operations.

### Connection Constraints

Connections must generate residuals rather than only copying outputs.

Fluid connection residuals should enforce:

- Upstream and downstream connected pressure compatibility, with optional
  pressure drop only inside components that model it.
- Enthalpy or total enthalpy propagation depending on component type.
- Mass flow continuity. Use a consistent sign convention:
  flow positive from component outlet into connected inlet.
- Domain consistency is already enforced by ports; preserve it.

Shaft connection residuals should enforce:

- Equal shaft speed at connected shaft ports.
- Torque balance at rotor states or algebraic shaft nodes.
- Sign convention: turbine drive torque positive into shaft, pump/load torque
  positive out of shaft, friction/load negative in rotor equation.

Thermal connection residuals should enforce:

- Equal interface temperature if the connection is conductive/contact-like.
- Equal and opposite heat flow.
- Optional thermal resistance component if a temperature drop is expected.

### Boundary Conditions

Boundary conditions should be modelled as boundary components where possible:

- `PressureBoundary`: fixed pressure and thermodynamic state, unknown mdot.
- `MassFlowBoundary`: fixed mdot and enthalpy, unknown pressure.
- `TankBoundary`: dynamic or quasi-steady tank state.
- `AmbientBoundary`: pressure schedule for nozzle outlet.
- `ShaftSpeedBoundary`: imposed speed for component rig tests.

Avoid relying on bare dictionary keys for physical coupling in production
engine models. Dictionary commands may remain as a convenience layer that sets
boundary component values.

## Nonlinear Solver Requirements

Implement robust nonlinear solve utilities in `atha/solver/nonlinear.py`.

### Inner Algebraic Solver

The transient RHS should:

1. Receive `t` and `X`.
2. Build or update command vector `U(t, X, telemetry)`.
3. Use previous successful `Z` as the initial guess.
4. Solve `Rz(t, X, Z, U) = 0`.
5. Return `dXdt(t, X, Z, U)`.

The inner solver must support:

- Damped Newton.
- Trust-region or line-search fallback.
- Finite-difference Jacobian initially.
- Optional sparse Jacobian pattern.
- Per-variable scaling.
- Per-residual scaling.
- Bounds or transforms for strictly positive variables.
- Warm-start storage.
- Diagnostic reports for failed solves.

### Steady-State Solver

The steady-state solver should solve a combined vector:

```text
Y = [X_free, Z, trim_controls]
R = [dXdt_free, Rz, trim_residuals, fixed_state_residuals]
```

It must allow users to choose which values are fixed and which are free. For
example, a user may fix chamber pressure and mixture ratio while freeing valve
positions and shaft speed, or fix valve positions and solve for the resulting
operating point.

### Residual Diagnostics

Every solver failure must report:

- Solver type and time.
- Norm of residual.
- Largest residuals by name, value, scale, and normalized value.
- Variables near bounds.
- NaN or infinite variables.
- Last successful timestep if transient.
- Suggested action where possible, such as "increase initial pump speed",
  "valve area is zero", or "map extrapolation error".

This is critical for agent-based development because otherwise failures will
be opaque.

## Controls Architecture

Create a new package `atha/controls`.

### Core Classes

Implement these classes:

```text
Signal
Sensor
SetpointSchedule
PIDController
Actuator
ControlSystem
```

### Sensor

A sensor reads a named state, algebraic variable, output, or derived value.
It should support:

- Gain and offset.
- First-order lag.
- Optional noise for Monte Carlo or robustness tests.
- Saturation.
- Failure modes later, but not in the first implementation.

### SetpointSchedule

Setpoints should support:

- Constant values.
- Step changes.
- Linear ramps.
- Piecewise-linear schedules.
- Smooth ramps such as cosine or cubic smoothstep.
- Event-triggered setpoint changes.

### PIDController

PID must include:

- `Kp`, `Ki`, `Kd`.
- Controller direction: direct or reverse acting.
- Output lower and upper limits.
- Integrator lower and upper limits.
- Anti-windup by clamping or back-calculation.
- Derivative on measurement by default.
- Derivative filter time constant.
- Optional sample time for discrete control.
- Continuous control mode for ODE integration.
- Deadband.
- Bias.
- Manual output.
- Bumpless transfer.

PID internal states should be part of `X` when they evolve continuously. For a
sampled controller, update it at explicit sample events and hold output between
samples.

### Actuator

Actuators translate controller output to physical component commands. Required
actuator features:

- Position command.
- Actual position state.
- First-order lag.
- Opening and closing rate limits.
- Hard min/max position.
- Optional deadband.
- Optional stiction/hysteresis later.

For valves, the physical valve flow equation must use actual position, not the
command position.

### Profile Integration

Extend `PhaseDefinition` to include:

- Open-loop command schedules.
- Controller enable/disable schedules.
- Setpoint schedules.
- Event definitions.
- Recording requests.

Existing `ControlCommand` can remain as a simple open-loop command but should
not be the only control mechanism.

## Component Physics Upgrade Plan

Upgrade components incrementally after the solver core can handle algebraic
unknowns and residuals.

### Volume

Current volume states `P,h` are a good choice. Improvements:

- Support liquid compressibility and two-phase detection explicitly.
- Replace ideal-gas-like pressure derivative with a formulation based on
  thermodynamic partial derivatives where available.
- Add optional heat transfer and wall contact ports.
- Enforce mass and energy conservation diagnostics.
- Handle reverse flow enthalpy correctly at each port.
- Add optional ullage/tank variants.

Acceptance tests:

- Constant inflow to closed volume increases pressure.
- Equal inflow/outflow at same enthalpy reaches steady state.
- Heat input raises enthalpy and temperature.
- Liquid volume pressure response matches bulk modulus approximation.

### Pipe And Line Components

Provide two families:

- Algebraic line: quasi-steady pressure drop.
- Dynamic line: inertance with mass flow state.

Improvements:

- Darcy friction factor as function of Reynolds number and roughness.
- Minor losses through K factors.
- Optional compressibility.
- Optional heat leak.
- Optional check valve behavior.
- Bidirectional flow with correct sign.

Acceptance tests:

- Pressure drop scales approximately with `mdot * abs(mdot)`.
- Dynamic pipe accelerates flow under pressure difference.
- Reverse pressure difference produces reverse flow if allowed.
- Closed check valve blocks reverse flow.

### Valve

Current valve is instantaneous, incompressible, and uses `A_frac` directly.
Improvements:

- Add `position` as a dynamic state.
- Add command-to-position actuator dynamics.
- Support CdA map versus position and pressure ratio.
- Support liquid valve equation.
- Support compressible gas valve equation with choking.
- Support leakage at closed position.
- Support cavitation/choked liquid option.
- Support rate limits and position limits.

Acceptance tests:

- Step command produces finite opening time.
- Position saturates at min/max.
- Flow follows CdA map.
- Compressible flow chokes at critical pressure ratio.
- Closed leakage is nonzero only if configured.

### Pump

Current pump map uses only speed-dependent pressure rise and constant
efficiency. Improvements:

- Use head coefficient maps versus corrected flow and speed.
- Solve pump operating point from system back pressure and flow.
- Include efficiency map.
- Compute outlet enthalpy rise:
  `h_out = h_in + hydraulic_work_per_kg / eta_loss_model` or an equivalent
  energy-consistent formulation.
- Compute torque from shaft power.
- Add cavitation/NPSH margin.
- Add minimum speed behavior and startup fallback.
- Allow inducer or boost pump later.

Acceptance tests:

- At design speed and design flow, map returns design head and efficiency.
- Pump power equals `mdot * delta_h_actual`.
- Torque times speed equals shaft power.
- Efficiency loss appears as fluid heating.
- Cavitation warning/limit triggers below NPSH requirement.

### Turbine

Current turbine uses a simple ideal-gas formula. Improvements:

- Use corrected flow map.
- Use pressure ratio and speed parameter maps.
- Use `ThermoBackend.isentropic_expansion()` for ideal exit state.
- Compute actual outlet enthalpy from efficiency.
- Compute torque from extracted power.
- Model choking.
- Support turbine bypass valve coupling.

Acceptance tests:

- Power equals `mdot * (h_in - h_out)`.
- Torque times speed equals power.
- Map efficiency changes with PR or speed ratio.
- Choked turbine flow becomes insensitive to downstream pressure.

### Combustion Chamber And Preburner

Current chamber stores `design_MR` but does not use it in the heat release
physics. Improvements:

- Compute mixture ratio from oxidizer and fuel inlet flows.
- Use Cantera equilibrium or cached equilibrium tables to compute product
  state versus pressure and MR.
- Support combustion efficiency and c-star efficiency.
- Support finite combustion time or residence lag.
- Support injector pressure drop and atomization-related coefficients.
- Track fuel-rich or oxidizer-rich preburner product properties.
- Allow film cooling and secondary injection streams.

Acceptance tests:

- MR changes product temperature and gamma.
- Zero propellant flow does not create heat release.
- Chamber mass and energy are conserved.
- Preburner outlet temperature can be trimmed by MR.

### Nozzle

Current nozzle computes choked mass flow and JANNAF-style thrust outputs.
Improvements:

- Make throat mass flow an algebraic relation coupled to chamber state.
- Support subsonic, choked, overexpanded, and separated regimes.
- Include ambient pressure schedules.
- Include throat erosion or area change later.
- Provide outputs: mdot, c-star, Cf, thrust, Isp, exit pressure, exit Mach,
  separation flag.

Acceptance tests:

- Choked flow is independent of ambient pressure until separation model applies.
- Thrust changes with ambient pressure.
- c-star relation matches `Pc * At / mdot`.
- Vacuum and sea-level thrust differ correctly.

### Rotor And Shaft Network

Current rotor state is useful but needs better coupling. Improvements:

- Support multiple drive and load ports.
- Enforce common speed algebraically for connected shaft ports.
- Include inertia, viscous friction, Coulomb friction optional, and external load.
- Support gear ratios later.
- Avoid artificial speed overrides except in explicit test boundaries.

Acceptance tests:

- Net positive torque accelerates shaft.
- Net negative torque decelerates shaft.
- Steady speed occurs when drive torque equals load plus friction.
- Pump and turbine powers balance at steady state.

### Regen Cooling

Current regen model is a single lumped channel. Improvements:

- Add segmented axial regen model with N stations.
- Track wall temperature per station.
- Track coolant enthalpy and pressure along the path.
- Use local hot-gas conditions along chamber/nozzle contour.
- Include roughness/friction, heat flux, and material temperature limits.
- Allow export of wall temperature profile.

Acceptance tests:

- Increasing coolant flow reduces wall temperature.
- Increasing chamber pressure increases heat flux.
- Outlet enthalpy rise equals integrated heat absorbed.
- Pressure drop increases with flow.

## Maps And Data

The existing `PerformanceMap` is a strong base. Improvements:

- Add named axis units and output units.
- Add map metadata: source, date, fluid, component, valid range.
- Add monotonicity checks where applicable.
- Add extrapolation policy per axis.
- Add corrected-speed/corrected-flow helpers.
- Add map plotting and validation utilities.
- Add CSV parsing fixes and tests for structured and scattered map loading.

Performance maps should be the primary way to introduce empirical behavior
without hard-coding cycle-specific equations.

## Profiles, Events, And Recording

Profiles should become the main way to define firing sequences.

### Required Phase Features

Each phase should support:

- Duration.
- Initial trim or inherited state.
- Open-loop command schedules.
- Controller setpoint schedules.
- Controller mode changes.
- Event-triggered end conditions.
- Hard abort limits.
- Soft warning limits.
- Recording rate and selected telemetry.
- Solver options.

### Event Handling

Implement event types:

- Time event.
- Variable crossing event.
- Limit event.
- Controller state event.
- User callback event.

When a command has a discontinuity, split the integration at the discontinuity
and restart the integrator with the latest state and algebraic warm start.

## Initialization Strategy

Real engine simulations are sensitive to initial guesses. Add explicit
initialization tools.

### Component Initialization

Every component should provide:

- Required operating point guesses.
- Default safe guesses.
- Variable bounds.
- Residual scales.
- A method to estimate algebraic unknowns from nearby states and commands.

### Cycle Initialization

Add `OperatingPoint` and `TrimProblem` classes:

```text
OperatingPoint:
    states
    algebraics
    commands
    outputs

TrimProblem:
    fixed_values
    free_controls
    targets
    bounds
    solve()
```

Support continuation:

1. Solve a simplified low-power point.
2. Increase target gradually.
3. Use each solution as the next initial guess.

This is essential for staged-combustion and full-flow cycles.

## Validation And Test Plan

Testing should be layered so failures identify the physical or numerical layer
that broke.

### Unit Tests

For every component:

- Constructor and variable registration.
- Units and scales.
- Limiting cases.
- Conservation checks.
- Reverse-flow or blocked-flow behavior.
- Map interpolation behavior.

### Solver Tests

Add tests for:

- Algebraic loop solve independent of component insertion order.
- Inner Newton warm start.
- Residual scaling.
- Diagnostics on impossible conditions.
- Bound handling.
- Transient restart at command discontinuity.

### Control Tests

Add tests for:

- PID proportional response.
- Integral accumulation.
- Anti-windup.
- Derivative filtering.
- Output saturation.
- Rate-limited actuator response.
- Closed-loop tracking of chamber pressure in a simple volume/nozzle model.

### Integration Tests

Add end-to-end tests:

- Pressure-fed engine steady and transient.
- Gas-generator shaft spin-up.
- Valve opening transient into chamber.
- Chamber pressure controller commanding propellant valves.
- Pump speed controller commanding shaft boundary in a rig.
- Regen channel thermal transient.
- ORSC and FFSC simplified cycles after DAE solver is stable.

### Validation Cases

Retain and expand TTBE validation. Add acceptance metrics:

- c-star error.
- Isp error.
- thrust error.
- pump outlet pressure error.
- turbine power balance error.
- transient timing and overshoot for representative steps.

## Implementation Phases

### Phase 1: Numerical Core

Goal: make the solver architecture physically correct before adding more
component detail.

Tasks:

1. Add variable and residual registries.
2. Add component variable registration API.
3. Add `EvaluationResult`.
4. Add layout-level `evaluate(t, X, Z, U)`.
5. Convert connection propagation into connection residuals.
6. Add nonlinear solver utility with scaling and diagnostics.
7. Implement transient inner algebraic solve.
8. Implement steady-state combined solve.
9. Add tests proving algebraic loops work.

Do not begin broad component upgrades until this phase is working.

### Phase 2: Boundary Components And Profiles

Goal: remove overreliance on raw boundary-condition dictionaries.

Tasks:

1. Add boundary components for pressure, mass flow, ambient, tank, and shaft
   speed.
2. Adapt profiles to set boundary component commands.
3. Add explicit schedule types: constant, step, ramp, piecewise, smoothstep.
4. Add integration segmentation at command discontinuities.
5. Preserve backwards compatibility where reasonable.

### Phase 3: Controls And Actuators

Goal: support closed-loop behavior.

Tasks:

1. Add `atha/controls`.
2. Implement sensor, setpoint, PID, actuator, and control system classes.
3. Add controller states to `X`.
4. Add actuator states to `X`.
5. Connect controller outputs to component commands.
6. Add closed-loop tests.

### Phase 4: Component Fidelity

Goal: upgrade physical models once the solver can support them.

Recommended order:

1. Valve with actuator dynamics.
2. Pump with real map behavior and enthalpy rise.
3. Turbine with corrected-flow map and thermo expansion.
4. Nozzle as algebraic choking component.
5. Chamber/preburner with MR-dependent Cantera cache.
6. Dynamic and algebraic pipe improvements.
7. Segmented regen.
8. Tank and pressurization components.

### Phase 5: Validation And Documentation

Goal: make the library usable and trustworthy.

Tasks:

1. Update README to describe the new DAE/control architecture.
2. Add detailed examples for pressure-fed, GG, ORSC, FFSC, and expander.
3. Add component rig examples for maps and validation.
4. Add diagnostic documentation.
5. Add validation reports with expected values and tolerances.

## Agent Guardrails

Agents implementing this plan should follow these rules.

1. Do not add a high-fidelity component that bypasses the global residual
   architecture.
2. Do not use component insertion order as a substitute for solving coupled
   equations.
3. Do not directly overwrite physical states to satisfy a controller target
   unless using an explicitly named idealized test mode.
4. Do not silently clamp unphysical values without reporting diagnostics.
5. Do not hide discontinuities inside smooth-looking solver calls; segment the
   integration or use events.
6. Do not introduce mixed units. All internals remain SI.
7. Do not make maps extrapolate silently by default for production simulations.
8. Do not remove simple examples; keep simple paths available for tests and
   teaching.
9. Preserve backwards compatibility where it does not compromise the numerical
   architecture.
10. Add focused tests with every behavior change.

## Definition Of Done

The implementation should be considered successful when ATHA can:

1. Build a nontrivial engine cycle from connected components.
2. Solve a steady trim point with algebraic loops independent of insertion
   order.
3. Run a transient with valves, shaft inertia, chamber/nozzle coupling, and
   changing ambient pressure.
4. Run a PID chamber-pressure controller that commands valve position through
   actuator dynamics.
5. Report clear diagnostics for impossible or poorly initialized solves.
6. Pass unit, solver, control, integration, and validation tests.
7. Demonstrate at least pressure-fed, gas-generator, staged-combustion, and
   expander examples with documented assumptions and limitations.

## Immediate Next Step

The next implementation task should be Phase 1: Numerical Core. In practical
terms, start by designing `VariableRegistry`, `ResidualRegistry`,
`EvaluationResult`, and a small proof-of-concept DAE solve with:

- One upstream pressure boundary.
- One valve or orifice.
- One chamber volume.
- One nozzle or downstream pressure boundary.

The proof should demonstrate that mass flow and chamber pressure are solved by
global residuals, not by propagation order. Once that works, extend the same
pattern to shaft torque balance and then to controls.
