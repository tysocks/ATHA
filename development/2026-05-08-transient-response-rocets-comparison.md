# Transient Response ROCETS Comparison

Date: 2026-05-08

## Sources Consulted

- Local ROCETS final report: `resources/19910011919.pdf`
- NASA NTRS page for NASA-CR-184099 / document ID 19910011919:
  https://ntrs.nasa.gov/citations/19910011919
- NASA NTRS RL10 ROCETS model record:
  https://ntrs.nasa.gov/citations/19950017370
- NASA NTRS IPD ROCETS transient simulation record:
  https://ntrs.nasa.gov/citations/20040075665

## ROCETS Transient Model Lessons

The ROCETS report is very clear that engine transients are not only schedules
applied to algebraic component maps. The important dynamic mechanisms are:

- volume capacitance: conservation of mass and energy using density/internal
  energy in ROCETS, often iterated through pressure/enthalpy variables;
- flow inertia: conservation of momentum with flow rate as a dynamic state;
- rotor inertia: shaft speed integration;
- thermal capacitance: metal/fluid thermal states;
- algebraic balances closed at the same time as transient corrector equations.

The report also describes state activation modes: ON, OFF, and STEADY-STATE.
This matters because a model can selectively retain or remove dynamic effects
during verification and reduced-order studies.

The public NTRS summary reinforces that ROCETS supported steady trim,
transient operation, linear partial generation, run/execution/output
processors, and trapezoidal/Gear corrector methods. Later public NASA records
for RL10 and IPD also frame ROCETS as a modular framework for steady-state and
transient propulsion-system simulation, validation against test/flight data,
large nonlinear solves, fluid properties, flow devices, and user constraints.

## ATHA Transient Response Gaps

### 1. Missing Global Algebraic Fluid Solve

ATHA can register connection residuals, but it does not yet solve a global
square algebraic network each transient step. This is the root cause of
non-ROCETS-like startup behavior in examples where valves, injectors, volumes,
and nozzles should interact through pressure and mass-flow constraints.

Required fix:

- Assemble port pressure, enthalpy, temperature, and flow unknowns into a
  global algebraic vector.
- Solve component residuals and connection residuals together at each RHS or
  implicit step.
- Use previous-step algebraic states as warm starts.

### 2. Flow Inertia Is Mostly Absent

ROCETS explicitly models pipe/inertial flow states. ATHA currently has many
algebraic flow elements but not a general reusable pipe-inertia component.

Required fix:

- Add a `PipeInertia` or `InertialFlowPath` component with `mdot` as a state.
- Use momentum balance:
  `dmdot/dt = (P_up - P_down - losses - elevation) / inertance`.
- Couple it to volume states so pressure and flow can overshoot.

### 3. Valve Transients Are Command Schedules, Not Component Dynamics

ATHA timings can step or ramp a valve command, but the valve itself does not
own a position state, actuator lag, rate limit, or saturation.

Required fix:

- Implement transient blocks from YAML:
  - first-order lag;
  - rate limiter;
  - min/max saturation;
  - optional deadband.
- Bind these blocks to component command/state paths such as
  `main_valve.position`.

### 4. Telemetry Should Be Processor-Owned

ATHA has begun moving toward a ROCETS-like Output Processor. TCA examples now
use telemetry YAML and `sample_rate_hz`. This should be extended to all
examples and to HDF5.

Required fix:

- Validate telemetry channel sources before the run.
- Export raw solver data and telemetry-resampled data separately.
- Add HDF5 output with metadata and units.

## Simple Demonstration Added

Example 15 was added:

```text
examples/15_valve_volume_transient/run.py
```

It uses:

- `timings.yaml` to step `valve.command` from closed to open at `t=0`;
- a first-order valve actuator state for actual `valve.position`;
- an upstream supply pressure;
- an isothermal downstream gas volume;
- a downstream outlet flow state driven by pressure drop, outlet resistance,
  and outlet inertance;
- telemetry YAML for exported pressure and mass-flow channels.

This intentionally demonstrates the simplest ROCETS transient lesson:
downstream pressure should not respond instantly to a valve command when there
is a storage volume. The pressure and outlet mass flow are separate solved
states, so the outlet flow now lags the pressure-driven steady value instead of
being a direct algebraic copy of pressure.

Current representative output:

```text
t=-1.0 s  cmd=0.0  valve=0.000  P=1.013 bar  mdot_in=0.00000  mdot_out=0.00000 kg/s
t= 0.0 s  cmd=1.0  valve=0.000  P=1.013 bar  mdot_in=0.00000  mdot_out=0.00000 kg/s
t= 0.5 s  cmd=1.0  valve=0.760  P=1.022 bar  mdot_in=0.00273  mdot_out=0.00002 kg/s
t= 1.0 s  cmd=1.0  valve=0.943  P=1.039 bar  mdot_in=0.00339  mdot_out=0.00009 kg/s
t=10.0 s  cmd=1.0  valve=1.000  P=1.259 bar  mdot_in=0.00359  mdot_out=0.00232 kg/s
```

The response is deliberately slow because the selected volume, actuator time
constant, outlet resistance, and outlet inertance create visible dynamic
separation.

## Transient YAML Runtime Added

The first reusable transient runtime has been added. Analysis YAML can now
reference a single transient library:

```yaml
transients: transients.yaml
```

The library supports these scalar response types:

- `table`
- `first_order`
- `second_order`
- `linear`
- `rate_limited`

`timings.yaml` remains the command script. `transients.yaml` defines how actual
component state responds to that command, for example:

```yaml
transients:
  valve_a_linear:
    type: linear
    input: valve_a.command
    output: valve_a.position
    initial: 0.2
    parameters:
      duration: 2.0
      from: 0.2
      to: 1.0
```

Example 16 demonstrates two valve trains feeding pipes, injectors, a chamber,
and a nozzle. Valve A uses the linear response above; valve B receives its open
command one second later and uses a first-order response.

## Next Implementation Steps

1. Promote the example-local outlet inertance equation into a reusable
   `PipeInertia` component and add a lower-damping variant with overshoot.
2. Bind transient blocks directly to component state registration so standard
   `TransientSolver` runs can own these actuator states without example-local
   glue code.
3. Implement global algebraic port solving for pressure/flow networks.
4. Convert example 14 from command-gated mass-flow injectors to pressure-driven
   valve/injector/nozzle coupling.
5. Add a ROCETS-derived benchmark case for volume plus inertial pipe transient
   response before attacking full TTBE-scale validation.
