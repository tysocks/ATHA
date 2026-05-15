# Worked Examples

These examples are current public entry points for the YAML runner.

## Pressure-Fed Transient

Examples 15-18 cover pressure-fed valve, pipe, injector, chamber, nozzle, and
controller workflows.

```powershell
python -m atha.cli examples\18_tca_mdot_controller\configs --output-dir outputs\example18
```

Look for:

- controller error telemetry;
- valve command and position response;
- chamber pressure and thrust response.

## FFSC Reduced DAE Acceptance

Example 19 is the staged-combustion acceptance case. It uses named phases,
sampled controllers, pump `phi`/`psi` maps, shutdown timing, linearization, and
an acceptance report.

```powershell
python -m atha.cli examples\19_ffsc_dae_acceptance\configs --output-dir outputs\example19
```

Artifacts:

- `ffsc_dae_acceptance.csv`;
- `ffsc_dae_acceptance.h5`;
- `ffsc_dae_acceptance.acceptance.json`;
- `ffsc_dae_acceptance.linearization.json`.

## Gas-Generator Single-Shaft Transient

Example 20 is the methalox gas-generator cycle with a PID-controlled generator
branch and map-backed pump head response.

```powershell
python -m atha.cli examples\20_gg_single_shaft_methalox\configs --output-dir outputs\example20
```

Useful telemetry:

- `SHAFT_RPM`;
- `LOX_PUMP_DP`, `METHANE_PUMP_DP`;
- `LOX_PUMP_PHI`, `METHANE_PUMP_PHI`;
- PID `P_TERM`, `I_TERM`, `D_TERM`, and `RAW_COMMAND`.

## Sweep And Monte Carlo

Generic sweep and Monte Carlo analyses wrap a base analysis and apply dotted
YAML path overrides. See tests for compact examples until a stable public
example folder is promoted.

Recommended pattern:

```yaml
analysis:
  type: sweep
  base_type: profile
  perturbations:
    sweep:
      - path: controllers.controllers.lox_generator_mdot_p.parameters.proportional_gain
        values: [0.1, 0.2, 0.3]
```

## Linearization

Use `analysis.type: linearization` for generic DAE linearization cases, or the
example 19 reduced FFSC linearization artifact while the full port-variable FFSC
path is still under development.

