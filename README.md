# ATHA - Advanced Transient and High-fidelity Analysis

ATHA is a Python toolkit for liquid rocket engine cycle modelling, transient
analysis, test-profile simulation, and uncertainty analysis. The architecture is
based on the NASA ROCETS approach: reusable component models are connected into
an engine topology, run conditions are supplied separately, and analysis/output
settings are controlled by a run manifest.

The current overhaul branch introduces a YAML-first configuration layer while
preserving the existing stable solver behavior.

## What Is Implemented

- Component graph modelling with typed fluid, shaft, and thermal ports.
- Steady-state trim with the existing Newton-based solver.
- Transient integration with SciPy Radau.
- Performance maps loaded from YAML references to constants, CSV files, or HDF5.
- Modular YAML files for engine topology, maps, boundaries, telemetry, and
  analysis settings.
- Monte Carlo and sweep analysis using existing runner infrastructure.
- Eight maintained YAML-driven examples:
  - `examples/04_gg_single_shaft_mc_sweep/run.py`
  - `examples/09_ffsc_lox_methane/run.py`
  - `examples/10_gg_lox_methane/run.py`
  - `examples/13_tca_pms_runbox/run.py`
  - `examples/14_tca_pms_valve_timing/run.py`
  - `examples/15_valve_volume_transient/run.py`
  - `examples/16_two_valve_transient_chain/run.py`
  - `examples/17_tca_propellant_valve_transient/run.py`

Legacy examples have been removed so these maintained examples are the canonical
starting points.

## Installation

Create a virtual environment with Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

If Python is not on `PATH`, use the full executable path:

```powershell
C:\Users\tyler\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Core dependencies include NumPy, SciPy, CoolProp, Cantera, PyYAML, h5py, joblib,
SALib, matplotlib, and pytest.

## Running Tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m pytest tests\unit -q
.venv\Scripts\python.exe -m pytest tests\integration -q
```

Current verified result on this branch:

```text
255 passed
```

## YAML Configuration Model

ATHA uses one top-level **Analysis YAML** as the run entrypoint. That file
references the other modular files.

```yaml
name: gg_lox_methane
engine: engine.yaml

maps:
  lox_pump_efficiency: maps/lox_pump_efficiency.yaml
  fuel_pump_efficiency: maps/fuel_pump_efficiency.yaml

boundary_conditions: boundaries.yaml
telemetry: telemetry.yaml

solver:
  steady_trim:
    tol: 1.0e-8
    max_iter: 200

analysis:
  type: nominal_mc_sweep
  speed_sweep:
    rpm_min: 15000
    rpm_max: 33000
    points: 15
```

### Engine YAML

The Engine YAML defines stable hardware structure:

- components
- component parameters
- map slots
- connections/layout

Example component with a map slot:

```yaml
components:
  lox_pump:
    type: Pump
    parameters:
      diameter: 0.085
      pump_map:
        mdot_design: 4.56
        dP_design: 15.1e6
        speed_design: 28000
        efficiency_design: 0.70
    maps:
      efficiency_map:
        ref: lox_pump_efficiency
        output: efficiency
```

Example connection:

```yaml
connections:
  - from: lox_pump.outlet
    to: lox_inj.inlet
    domain: fluid
```

### Map YAML

Map YAML files define where map data comes from. Small maps can be constants;
larger maps should reference CSV or HDF5 data exported from tests, simulations,
or component sweeps.

Constant map:

```yaml
name: lox_pump_efficiency
kind: constant
source:
  type: constant
  values:
    efficiency: 0.70
outputs:
  - name: efficiency
```

CSV map:

```yaml
name: pump_combo
kind: structured_grid
source:
  type: csv
  path: data/pump_combo.csv
axes:
  - name: corrected_speed
    column: Nc
  - name: corrected_flow
    column: Wc
outputs:
  - name: head
    column: head_j_per_kg
  - name: efficiency
    column: eta
interpolation:
  extrapolation: clamp
```

Multiple component map slots can reference the same multi-output map:

```yaml
maps:
  head_map:
    ref: pump_combo
    output: head
  efficiency_map:
    ref: pump_combo
    output: efficiency
```

### Transients YAML

Transient YAML files define actual component response to commanded values. The
test script should command paths in `timings.yaml`; the transient library
defines how the hardware state moves.

```yaml
transients: transients.yaml
```

```yaml
name: actuator_transients
transients:
  main_valve:
    type: first_order
    input: main_valve.command
    output: main_valve.position
    initial: 0.0
    parameters:
      time_constant: 0.35
      lower_limit: 0.0
      upper_limit: 1.0
```

Supported scalar response types are `table`, `first_order`, `second_order`,
`linear`, and `rate_limited`.

### Boundary Conditions YAML

Boundary conditions are imposed physical values from the test stand or
environment. They can be constant or time-varying.

```yaml
name: boundaries
time_unit: s
conditions:
  lox_tank.outlet.P: {value: 4.0e5, units: Pa}
  lox_tank.outlet.T: {value: 91.0, units: K}
  nozzle.ambient.P: {value: 0.0, units: Pa}
  shaft.omega_override:
    schedule:
      type: table
      values:
        - [0.0, 1000.0]
        - [1.0, 2300.0]
```

Supported schedule types are `constant`, `step`, `ramp`, and `table`.

### Telemetry YAML

Telemetry files define the channels to export or plot.

```yaml
name: telemetry
sample_rate_hz: 100
channels:
  - alias: PC
    source: chamber.P
    units: MPa
  - alias: THRUST
    source: nozzle.thrust
    units: N
exports:
  csv: true
  hdf5: true
```

## Running The Examples

Each maintained example is self-contained under `examples/<analysis>/`.
The Python entrypoint is `run.py`; YAML files live in that analysis folder's
`configs/` directory.

```powershell
.venv\Scripts\python.exe examples\04_gg_single_shaft_mc_sweep\run.py
.venv\Scripts\python.exe examples\09_ffsc_lox_methane\run.py
.venv\Scripts\python.exe examples\10_gg_lox_methane\run.py
.venv\Scripts\python.exe examples\13_tca_pms_runbox\run.py
.venv\Scripts\python.exe examples\14_tca_pms_valve_timing\run.py
.venv\Scripts\python.exe examples\15_valve_volume_transient\run.py
.venv\Scripts\python.exe examples\16_two_valve_transient_chain\run.py
.venv\Scripts\python.exe examples\17_tca_propellant_valve_transient\run.py
```

Outputs are written to `outputs/`, including Monte Carlo HDF5 results and sweep
plots where applicable.

### Example 04

`examples/04_gg_single_shaft_mc_sweep/run.py`

Gas-generator LOX/ethanol cycle with one turbopump shaft. Runs:

- nominal steady trim
- pump/GG uncertainty Monte Carlo
- shaft-speed sweep

Config directory:

```text
examples/04_gg_single_shaft_mc_sweep/configs/
```

### Example 09

`examples/09_ffsc_lox_methane/run.py`

Full-flow staged-combustion LOX/methane cycle with two shafts, preburners, a
regen channel, and a startup profile.

Config directory:

```text
examples/09_ffsc_lox_methane/configs/
```

### Example 10

`examples/10_gg_lox_methane/run.py`

Gas-generator LOX/methane cycle with one shaft and regenerative cooling. Runs:

- nominal steady trim
- pump/GG uncertainty Monte Carlo
- shaft-speed sweep

Config directory:

```text
examples/10_gg_lox_methane/configs/
```

### Example 13

`examples/13_tca_pms_runbox/run.py`

Simple thrust chamber assembly with LOX mass-flow injector, methane mass-flow
injector, chamber, and nozzle. It consumes a time-tagged PMS target profile of
total mass flow and OF. The current example profile traces a runbox around the
nominal operating point from example 09:

```text
mdot_lox   = 4.23 kg/s
mdot_fuel  = 1.21 kg/s
mdot_total = 5.44 kg/s
OF         = 3.50
```

The default runbox traces the perimeter of:

```text
mdot_total = 80-120% of setpoint
OF         = 85-115% of setpoint
```

The PMS profile is defined in:

```text
examples/13_tca_pms_runbox/configs/operating_conditions.yaml
```

The time-tagged target profile is loaded from a separate data file referenced by
that YAML:

```yaml
schedule:
  type: profile
  source:
    type: json
    path: data/pms_runbox_targets.json
  time_column: time_s
  outputs:
    mdot_total: mdot_total
    OF: OF
```

The JSON file contains test-style targets:

```json
{
  "targets": [
    {"time_s": 0.0, "mdot_total": 4.352, "OF": 2.971515},
    {"time_s": 0.25, "mdot_total": 4.662857, "OF": 2.971515}
  ]
}
```

CSV target profiles are also supported:

```yaml
schedule:
  type: profile
  source:
    type: csv
    path: data/pms_targets.csv
  time_column: time_s
  outputs:
    mdot_total: mdot_total
    OF: OF
```

Paths are resolved relative to the operating-conditions YAML file.

The injector commands are defined with null controllers in:

```text
examples/13_tca_pms_runbox/configs/controllers.yaml
```

In this context, null controllers pass `pms.mdot_total` and `pms.OF` through to
command aliases. The `of_mass_flow_split` controller then calculates methane and
LOX injector mass flows from total mass flow and OF.

Outputs:

```text
outputs/tca_pms_runbox.csv
outputs/tca_pms_runbox.png
```

Config directory:

```text
examples/13_tca_pms_runbox/configs/
```

### Example 14

`examples/14_tca_pms_valve_timing/run.py`

Copy of the simple TCA PMS profile with upstream LOX and methane valves added
ahead of the injectors. The PMS target profile still provides `mdot_total` and
`OF`; the run starts with a two-second closed-valve dwell from `t=-2 s` to
`t=0 s`. `timings.yaml` then ramps both valves open from `t=0 s` to `t=0.75 s`.
The controller calculates LOX/methane mass-flow commands and gates them by the
timed valve positions.

Timing file:

```text
examples/14_tca_pms_valve_timing/configs/timings.yaml
```

Outputs:

```text
outputs/tca_pms_valve_timing.csv
outputs/tca_pms_valve_timing.png
```

### Example 15

`examples/15_valve_volume_transient/run.py`

Minimal transient response demonstration: a timed valve opens into a downstream
gas volume with actuator lag and an outlet flow-inertia state. The valve command
comes from `timings.yaml`; actual valve position, downstream pressure, and
inlet/outlet mass flow are exported from `telemetry.yaml`. Pressure and outlet
mass flow are solved as separate dynamic states, so the outlet flow no longer
has the same normalized rise as downstream pressure.

Outputs:

```text
outputs/valve_volume_transient.csv
outputs/valve_volume_transient.png
```

### Example 16

`examples/16_two_valve_transient_chain/run.py`

Two fixed-supply valve trains feeding pipes, injectors, a chamber, and a
nozzle. Both valve commands start at 20 percent. Valve A uses a linear transient
definition and opens over two seconds from `t=0`; valve B uses a first-order
transient definition and receives its open command one second later. The example
demonstrates `transients.yaml` as the hardware response layer separate from
`timings.yaml` command events.

Outputs:

```text
outputs/two_valve_transient_chain.csv
outputs/two_valve_transient_chain.png
```

### Example 17

`examples/17_tca_propellant_valve_transient/run.py`

Methane/LOX thrust chamber transient with one methane valve and one LOX valve
upstream of separate injectors feeding a shared chamber and nozzle. Both valves
start at 20 percent. The methane valve uses a linear transient and opens over
two seconds from `t=0`; the LOX valve receives its open command at `t=1 s` and
follows a first-order transient response.

Outputs:

```text
outputs/tca_propellant_valve_transient.csv
outputs/tca_propellant_valve_transient.png
```

## Programmatic Loading

The config layer can be used directly:

```python
from atha.config import (
    build_performance_maps,
    evaluate_boundary_conditions,
    load_analysis_config,
)

loaded = load_analysis_config("examples/10_gg_lox_methane/configs/analysis.yaml")
maps = build_performance_maps(loaded.maps)
bcs_at_start = evaluate_boundary_conditions(loaded.boundary_conditions, 0.0)
```

The returned object contains:

- `loaded.analysis_config`
- `loaded.engine`
- `loaded.maps`
- `loaded.transients`
- `loaded.boundary_conditions`
- `loaded.operating_conditions`
- `loaded.timings`
- `loaded.controllers`
- `loaded.telemetry`

## Current Solver Notes

The examples intentionally use the current stable solver settings:

- steady trim via `SteadyStateSolver`
- transient integration via Radau where profiles are used
- solver tolerances configured in Analysis YAML

The codebase still contains the DAE foundation, but the full global
port-variable algebraic solve is not complete yet. Until that lands, examples
may include explicit design flow or shaft override boundary conditions to keep
the compatibility solver well-conditioned.

## References

- NASA ROCETS report: `resources/19910011919.pdf`
- JANNAF rocket engine performance reference: `resources/JANNAF ROCKET ENGINE.pdf`
