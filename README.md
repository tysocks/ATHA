# ATHA - Advanced Transient and High-fidelity Analysis

ATHA is a Python toolkit for liquid rocket engine cycle modelling, transient
analysis, test-profile simulation, and uncertainty analysis. The architecture is
inspired by NASA ROCETS: reusable components are connected into an engine
topology, boundary conditions and operating targets are supplied separately, and
the run/output configuration is controlled by YAML.

The current overhaul branch is YAML-first. Most user workflows should start
from a config folder and run through `run_config_folder` or `atha-run`.

## Current Status

Implemented:

- modular YAML loading for engine, maps, boundaries, operating conditions,
  controllers, timings, transients, telemetry, and analysis settings;
- component source catalogs and telemetry validation;
- transient valve response types: `table`, `first_order`, `second_order`,
  `linear`, `rate_limited`;
- pressure-fed TCA transient examples;
- reduced-order FFSC DAE acceptance example;
- finite-difference linearization artifacts;
- Monte Carlo and speed-sweep registry runner for gas-generator examples;
- CSV, HDF5, plot, manifest, residual, linearization, and acceptance outputs.

Still pending:

- full arbitrary ROCETS-like port DAE solve;
- full per-port pressure, enthalpy/temperature, density, and mass-flow histories;
- automatic connection residual assembly as the universal solver path;
- map-backed pump/turbine residuals in the arbitrary port solve.

## Installation

Create and install into a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

If Python is not on `PATH`:

```powershell
C:\Users\tyler\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Running

Run a config folder with the CLI:

```powershell
.venv\Scripts\python.exe -m atha.cli examples\19_ffsc_dae_acceptance\configs
```

or, after editable install:

```powershell
atha-run examples\19_ffsc_dae_acceptance\configs
```

Programmatic API:

```python
from atha.runner import run_config_folder

result = run_config_folder("examples/18_tca_mdot_controller/configs")
summary = result.require_summary()
print(summary.csv)
```

Outputs are written to `outputs/` by default.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\unit -q
.venv\Scripts\python.exe -m pytest tests\unit\test_config.py tests\unit\test_solver.py -q
```

Runtime markers are declared for `fast`, `integration`, `acceptance`,
`monte_carlo`, and `slow`.

## Documentation

- [YAML configuration guide](docs/configuration.md)
- [Path grammar](docs/path_grammar.md)
- [Adding models and outputs](docs/contributing_models.md)
- [Solver diagnostics](docs/solver_diagnostics.md)
- [Validation matrix](development/validation_matrix.md)
- [Architecture tracker](development/2026-05-08-general-config-runner-architecture.md)

## Example Categories

Runnable registry examples:

- `examples/04_gg_single_shaft_mc_sweep`
- `examples/10_gg_lox_methane`
- `examples/15_valve_volume_transient`
- `examples/16_two_valve_transient_chain`
- `examples/17_tca_propellant_valve_transient`
- `examples/18_tca_mdot_controller`
- `examples/19_ffsc_dae_acceptance`

Acceptance example:

- `examples/19_ffsc_dae_acceptance`

Compatibility examples retained for current solver behavior:

- `examples/09_ffsc_lox_methane`
- `examples/13_tca_pms_runbox`

## Maintained Examples

### Example 04: GG Single-Shaft MC Sweep

Gas-generator LOX/ethanol cycle with nominal steady trim, Monte Carlo, and
shaft-speed sweep.

```powershell
.venv\Scripts\python.exe examples\04_gg_single_shaft_mc_sweep\run.py
```

### Example 09: FFSC LOX/Methane Compatibility Case

Full-flow staged-combustion LOX/methane compatibility example using the older
explicit runner path. This remains useful for reference but is not yet the
arbitrary port DAE runner.

### Example 10: GG LOX/Methane

Gas-generator LOX/methane with regen, nominal trim, Monte Carlo, and speed
sweep.

```powershell
.venv\Scripts\python.exe examples\10_gg_lox_methane\run.py
```

### Example 13: TCA PMS Runbox Compatibility Case

Simple TCA PMS profile consuming a target file of total mass flow and OF. This
is retained as a compatibility workflow while the universal runner matures.

### Examples 15-18: Pressure-Fed Transient And Control

Examples 15-18 demonstrate valve/volume transients, two-valve chains,
methalox TCA valve response, and proportional mass-flow control.

```powershell
.venv\Scripts\python.exe examples\15_valve_volume_transient\run.py
.venv\Scripts\python.exe examples\16_two_valve_transient_chain\run.py
.venv\Scripts\python.exe examples\17_tca_propellant_valve_transient\run.py
.venv\Scripts\python.exe examples\18_tca_mdot_controller\run.py
```

### Example 19: FFSC DAE Acceptance

Reduced-order FFSC DAE acceptance case. This is the current acceptance gate for
the YAML architecture and the stepping stone to the full arbitrary port solve.

```powershell
.venv\Scripts\python.exe examples\19_ffsc_dae_acceptance\run.py
```

Key outputs:

- `outputs/ffsc_dae_acceptance.csv`
- `outputs/ffsc_dae_acceptance.h5`
- `outputs/ffsc_dae_acceptance.png`
- `outputs/ffsc_dae_acceptance.linearization.json`
- `outputs/ffsc_dae_acceptance.acceptance.json`

## References

- NASA ROCETS report: `resources/19910011919.pdf`
- JANNAF rocket engine performance reference: `resources/JANNAF ROCKET ENGINE.pdf`
