# ATHA verification guide

This guide explains how to run the Workstream 6.2 verification ladder and
interpret the generated artifacts.

## Quick start

```bash
# Level 0 unit/math checks
pip install -e ".[test]"
pytest tests/test_level0_reference_checks.py -q

# Level 2 subsystem suite (fast, ~minutes)
python examples/21_generic_port_subsystems/run_verification_suite.py
pytest tests/test_verification_subsystems.py -q

# Level 3 canonical engine gate (slow, ~2 minutes)
pytest tests/test_verification_engine.py -m slow -q
```

## Artifact types

| File pattern | Format | Meaning |
| --- | --- | --- |
| `*.acceptance.json` | `atha.acceptance_report.v1` | Pass/fail checks on telemetry arrays |
| `*.regression.json` | `atha.regression_report.v1` | Metric windows on CSV channels |
| `*.reference_checks.json` | `atha.reference_check_report.v1` | Analytical oracle comparisons |
| `verification_suite_report.json` | `atha.verification_suite_report.v1` | Batch summary for subsystem suite |
| `*.parity_report.json` | `atha.parity_report.v1` | Reference vs candidate overlays |

## Configuring acceptance in YAML

```yaml
analysis:
  acceptance:
    report: my_case.acceptance.json
    case: my_case_id
    evaluation_end_s: 24.99          # optional window cutoff
    require_solver_source: generic_port
    shaft_paths: [shaft.rpm]
    required_paths: [mdot.total, nozzle.thrust]
    tolerances:
      final_mdot_rel: 0.35
      mdot_tracking_rms_rel: 0.35
      min_peak_thrust: 100000.0
      max_normalized_residual: 1.0e6
```

Acceptance runs automatically after `analysis.type: profile` when the block is present.

## Programmatic suite execution

```python
from pathlib import Path
from atha.validation.verification_suite import run_verification_case, verification_cases

spec = next(case for case in verification_cases() if case.id == "regen_channel")
result = run_verification_case(spec, output_dir=Path("outputs/verification/regen_channel"))
assert result.passed
```

## Analytical reference checks

`atha.validation.reference_checks` exposes closed-form helpers:

- `orifice_mdot` — incompressible valve/orifice flow
- `nozzle_thrust` — Cf·At·(Pc−Pa)
- `characteristic_velocity` — ideal-gas c*
- `pump_head_affinity` / `pump_flow_affinity` — map scaling laws
- `regen_wall_temperature_rise` — lumped wall thermal inertia

Use these in notebooks, tests, or custom comparison scripts when a full
external solver export is unavailable.

## Parity mode (reference vs candidate)

Parity analysis is registered as `analysis.type: parity` and uses
`atha.validation.parity`. No retained example enables it yet; add a small case
when a frozen reference CSV is available.

## Adding a new verification case

1. Create a config folder with `analysis.yaml`, telemetry, and acceptance block.
2. Add a `VerificationCaseSpec` entry in `atha/validation/verification_suite.py`.
3. Optionally add a reference oracle under `verification/references/`.
4. Extend `tests/test_verification_subsystems.py` or create a focused test module.
5. Document the case purpose in `docs/VERIFICATION_MATRIX.md`.

## Canonical engine mdot tracking note

The FFSC acceptance case measures total propellant flow as the **sum of pump inlet
flows** when no explicit `mdot.total` algebraic variable exists. Nozzle throat
flow alone is insufficient for crossover-controller tracking in staged-combustion
topologies with bleeds and branch splits.
