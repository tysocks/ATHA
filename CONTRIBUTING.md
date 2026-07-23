# Contributing to ATHA

This guide covers the supported contributor workflows after Workstream 6.3.

## Canonical development path

All new work should target:

```text
YAML configs
  -> config.loader
  -> EngineAssembler / PortNetworkBuilder
  -> residual + derivative contracts
  -> DAEExecutionProblem
  -> outputs / acceptance / verification
```

Do **not** extend:

- `atha.components.factory.build_component_from_config`
- `atha.solver.steady_state.SteadyStateSolver`
- `atha.solver.transient.TransientSolver`
- Legacy `EngineLayout` solvers under `atha.solver`

See `docs/ARCHITECTURE.md` for the full package map.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Tests and verification

```bash
# Fast unit + subsystem gates
.venv/bin/python -m pytest tests/test_level0_reference_checks.py tests/test_quality_and_benchmarks.py -q
.venv/bin/python -m pytest tests/test_verification_subsystems.py -q

# Full-engine gate (slow)
.venv/bin/python -m pytest tests/test_verification_engine.py -m slow -q

# Subsystem suite runner
.venv/bin/python examples/21_generic_port_subsystems/run_verification_suite.py
```

Details: `docs/VERIFICATION_GUIDE.md` and `docs/VERIFICATION_MATRIX.md`.

## Lint / format

Ruff is the baseline formatter and linter:

```bash
.venv/bin/ruff check atha tests scripts
.venv/bin/ruff format --check atha tests scripts
```

To auto-fix:

```bash
.venv/bin/ruff check --fix atha tests scripts
.venv/bin/ruff format atha tests scripts
```

Type-checking (mypy/pyright) is intentionally deferred until the network/config
interfaces settle further.

## Benchmarks

```bash
.venv/bin/python scripts/run_benchmarks.py
.venv/bin/python scripts/run_benchmarks.py --include-slow
```

Reports land under `outputs/benchmarks/benchmark_report.json` with wall time,
algebraic solve counts, skipped second-pass solves, residual norms, and output
sizes.

## Adding a component

1. Implement a residual contract in `atha/components/residuals.py`.
2. Add a derivative contract in `atha/components/derivatives.py` when the type owns dynamic states.
3. Register the type in `default_component_registry()` (`atha/components/registry.py`).
4. Add or extend a subsystem gate under `examples/21_generic_port_subsystems/`.
5. Document maturity in `docs/COMPONENT_MATURITY.md` and verification status in `docs/VERIFICATION_MATRIX.md`.

## Adding a controller

1. Prefer existing controller types in `controllers.yaml` (`proportional`, `pi`, `pid`, `rate_limiter`, …).
2. Use mission-phase fields from `atha.config.mission_phases`:
   - `active_phases` / `inactive_phases`
   - `reset_on_enter`
   - `hold_when_inactive`
3. Keep controller evaluation order explicit when outputs feed other controllers.
4. Add telemetry for command/error channels and acceptance thresholds.

## Adding a verification case

1. Create a config folder with `analysis.yaml`, telemetry, and an `acceptance` block.
2. Register a `VerificationCaseSpec` in `atha/validation/verification_suite.py`.
3. Optionally add a design-point oracle under `verification/references/` or a historical
   package under `verification/historical/` (see `docs/HISTORICAL_CORRELATION.md`).
4. Extend `tests/test_verification_subsystems.py` or a focused module.
5. Document the case in `docs/VERIFICATION_MATRIX.md`.

For external CSV parity, prefer `analysis.type: parity` with `parity.reference_csv`
(see `examples/24_pump_map_historical_parity`).

## Documentation map

| Doc | Purpose |
| --- | --- |
| `README.md` | Install, run, scope |
| `docs/ARCHITECTURE.md` | Canonical vs legacy paths |
| `docs/COMPONENT_MATURITY.md` | Residual/derivative maturity |
| `docs/VERIFICATION_GUIDE.md` | How to run verification |
| `docs/VERIFICATION_MATRIX.md` | Component/case status |
| `IMPLEMENTATION_PLAN.md` | Roadmap and workstream status |
| `CONTRIBUTING.md` | This file |
