# ATHA - Advanced Transient and High-fidelity Analysis

ATHA is a YAML-driven liquid rocket engine cycle and transient simulation
toolkit inspired by ROCETS. The retained project tree focuses on the generic
DAE runner and active cycle examples 19–23.

## Current scope

ATHA currently provides:

- config-driven assembly of liquid-engine port networks;
- steady / profile / linearization / parity analysis modes;
- mission-phase timing with phase-aware controllers;
- residual/derivative contracts for the major engine components;
- acceptance, regression, parity, and verification-suite outputs;
- a lightweight runtime benchmark suite.

It is strongest today as a **system-level mission-cycle framework**. Component
fidelity and external historical-data correlation continue under later
workstreams; see `IMPLEMENTATION_PLAN.md`.

## Canonical execution path

```text
YAML config folder
  -> atha.config.loader
  -> atha.runner.run_config_folder / SolverDriver
  -> EngineAssembler + PortNetworkBuilder
  -> DAEExecutionProblem (phases, controllers, residuals, derivatives)
  -> telemetry / acceptance / regression / parity
```

Details: `docs/ARCHITECTURE.md`.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

`[dev]` includes pytest and Ruff. For runtime-only installs use `pip install -e .`.

## Run

Retained examples:

```bash
.venv/bin/python examples/19_ffsc_dae_acceptance/run.py
.venv/bin/python examples/20_gg_single_shaft_methalox/run.py
.venv/bin/python examples/22_ethanol_lox_5kn_two_shaft_gg/run.py
.venv/bin/python examples/23_single_lox_pump_map/run.py
```

CLI:

```bash
.venv/bin/python -m atha.cli examples/21_generic_port_subsystems/chamber_nozzle --progress
.venv/bin/python -m atha.cli examples/21_generic_port_subsystems/regen_channel --progress
```

Programmatic API:

```python
from atha.runner import run_config_folder

result = run_config_folder("examples/20_gg_single_shaft_methalox/configs", progress=True)
summary = result.require_summary()
print(summary.csv)
```

Outputs default to `outputs/` unless an alternate directory is supplied.

## Analysis modes and outputs

| Mode | Purpose | Typical artifacts |
| --- | --- | --- |
| `profile` | Transient mission / subsystem profile | CSV, PNG, diagnostics JSON, acceptance JSON |
| `steady` | Algebraic trim / steady network solve | diagnostics JSON |
| `linearization` | Finite-difference state-space snapshot | linearization JSON |
| `parity` | Reference vs candidate telemetry compare | parity report + delta CSV |

Acceptance / regression blocks are configured under `analysis:` in YAML and write
JSON reports next to telemetry.

## Retained examples

| Example | Role |
| --- | --- |
| `examples/19_ffsc_dae_acceptance` | **Canonical** FFSC full mission-cycle case |
| `examples/20_gg_single_shaft_methalox` | Single-shaft GG + PID template |
| `examples/21_generic_port_subsystems` | Fast subsystem / MVP verification gates |
| `examples/22_ethanol_lox_5kn_two_shaft_gg` | Two-shaft GG mission-profile template |
| `examples/23_single_lox_pump_map` | Pump-map transient verification seed |
| `examples/24_pump_map_historical_parity` | Historical/literature parity vs affinity oracle |

## Tests, lint, and benchmarks

```bash
# Verification gates
.venv/bin/python -m pytest tests/test_level0_reference_checks.py tests/test_verification_subsystems.py -q
.venv/bin/python -m pytest tests/test_verification_engine.py -m slow -q

# Lint / format
.venv/bin/ruff check atha tests scripts
.venv/bin/ruff format --check atha tests scripts

# Runtime benchmarks (fast + medium by default)
.venv/bin/python scripts/run_benchmarks.py

# Historical / external correlation suite
.venv/bin/python scripts/run_historical_correlation.py
```

## Documentation

- `IMPLEMENTATION_PLAN.md` — roadmap and Workstream status
- `CONTRIBUTING.md` — how to add components, controllers, verification cases
- `docs/ARCHITECTURE.md` — canonical package map and legacy boundaries
- `docs/COMPONENT_MATURITY.md` — residual/derivative maturity matrix
- `docs/CANONICAL_MISSION_CASE.md` — example 19 mission-cycle definition
- `docs/MISSING_PHYSICS_BACKLOG.md` — prioritized physics gaps
- `docs/VERIFICATION_GUIDE.md` — how to run acceptance / verification
- `docs/VERIFICATION_MATRIX.md` — component and case verification status
- `docs/HISTORICAL_CORRELATION.md` — external/historical data ingestion and metrics
- `docs/reports/FFSC_CANONICAL_VERIFICATION_REPORT.md` — Level-3 engine report
- `docs/reports/HISTORICAL_CORRELATION_REPORT.md` — Workstream 6.4 correlation report
