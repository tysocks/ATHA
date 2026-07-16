# ATHA - Advanced Transient and High-fidelity Analysis

ATHA is a YAML-driven liquid rocket engine cycle and transient simulation
toolkit inspired by ROCETS. The retained project tree is focused on the generic
DAE runner and the active cycle examples.

## Current scope

ATHA currently provides:

- config-driven assembly of liquid-engine port networks;
- steady / profile / linearization / parity analysis modes;
- mission-phase timing with phase-aware controllers;
- residual/derivative contracts for the major engine components;
- example-driven acceptance, regression, and telemetry outputs.

It is strongest today as a **system-level mission-cycle framework**. Component
fidelity and external verification are actively being improved; see
`IMPLEMENTATION_PLAN.md` and `docs/`.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Run

Run a retained example directly:

```bash
.venv/bin/python examples/19_ffsc_dae_acceptance/run.py
.venv/bin/python examples/20_gg_single_shaft_methalox/run.py
.venv/bin/python examples/22_ethanol_lox_5kn_two_shaft_gg/run.py
.venv/bin/python examples/23_single_lox_pump_map/run.py
```

Or run any retained config folder through the CLI:

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

Outputs are written to `outputs/` by default unless an alternate output
directory is supplied.

## Retained Examples

| Example | Role |
| --- | --- |
| `examples/19_ffsc_dae_acceptance` | **Canonical** FFSC full mission-cycle case |
| `examples/20_gg_single_shaft_methalox` | Single-shaft GG + PID template |
| `examples/21_generic_port_subsystems` | Fast subsystem / MVP gates |
| `examples/22_ethanol_lox_5kn_two_shaft_gg` | Two-shaft GG mission-profile template |
| `examples/23_single_lox_pump_map` | Pump-map transient verification seed |

Examples 19–23 run through the generic-port DAE path. Historical runner
alternatives have been removed from the retained project tree; run provenance,
acceptance reports, and direct `run.py` output include `solver_source` as a
generic-port guardrail.

## Documentation

- `IMPLEMENTATION_PLAN.md` — roadmap and Workstream status
- `docs/ARCHITECTURE.md` — canonical package map and legacy boundaries
- `docs/COMPONENT_MATURITY.md` — component residual/derivative maturity matrix
- `docs/CANONICAL_MISSION_CASE.md` — example 19 mission-cycle definition
- `docs/MISSING_PHYSICS_BACKLOG.md` — prioritized physics gaps
