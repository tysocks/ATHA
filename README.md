# ATHA - Advanced Transient and High-fidelity Analysis

ATHA is a YAML-driven liquid rocket engine cycle and transient simulation
toolkit inspired by ROCETS. The retained project tree is focused on the generic
DAE runner and the active cycle examples.

## Install

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

If Python is not on `PATH`:

```powershell
C:\Users\tyler\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

## Run

Run a retained example directly:

```powershell
.venv\Scripts\python.exe examples\19_ffsc_dae_acceptance\run.py
.venv\Scripts\python.exe examples\20_gg_single_shaft_methalox\run.py
.venv\Scripts\python.exe examples\22_ethanol_lox_5kn_two_shaft_gg\run.py
```

Or run any retained config folder through the CLI:

```powershell
.venv\Scripts\python.exe -m atha.cli examples\21_generic_port_subsystems\chamber_nozzle --progress
```

The direct `run.py` files for examples 19, 20, and 22 enable live solver
progress automatically. CLI runs can use `--progress` or `--no-progress`.

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

- `examples/19_ffsc_dae_acceptance`
- `examples/20_gg_single_shaft_methalox`
- `examples/21_generic_port_subsystems`
- `examples/22_ethanol_lox_5kn_two_shaft_gg`

Examples 19, 20, and 22 run through the generic-port DAE path. Historical
runner alternatives have been removed from the retained project tree; run
provenance, acceptance reports, and direct `run.py` output include
`solver_source` as a generic-port guardrail.
