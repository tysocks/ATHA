# Example 19: Canonical FFSC Mission-Cycle Acceptance Case

This is the **canonical ATHA full-engine mission-cycle case** selected under
Workstream 6.1.

It runs the full-flow staged-combustion methalox architecture through the
generic-port DAE profile solver (`solver_source: generic_port`) and exercises:

- dual turbopump shafts with pump and turbine affinity maps;
- four preburner-branch valves plus main propellant valves;
- transient valve response definitions;
- operating-condition targets with CLC, throttle, recover, and rundown;
- phase-specific proportional controllers;
- telemetry export, linearization, and acceptance reporting.

## Mission phases

| Phase | Window | Control mode |
| --- | --- | --- |
| `prestart` | 0–1 s | open-loop |
| `startup` | 1–3 s | open-loop |
| `CLC` | 3–10 s | closed-loop |
| `throttle` | 10–20 s | closed-loop throttle |
| `CLC_recover` | 20–25 s | closed-loop recover |
| `shutdown` | 25–28 s | open-loop valve close |
| `rundown` | 28–30 s | open-loop coast |

Controllers are active only in `CLC`, `throttle`, and `CLC_recover`, with
`reset_on_enter: [CLC]`. `hold_when_inactive` is disabled here because shutdown
valve motion is owned by `timings.yaml`.

## Outputs

- `outputs/ffsc_dae_acceptance.csv`
- `outputs/ffsc_dae_acceptance.h5`
- `outputs/ffsc_dae_acceptance.png`
- `outputs/ffsc_dae_acceptance.linearization.json`
- `outputs/ffsc_dae_acceptance.acceptance.json`

## Notes

- Regen cooling is intentionally not yet part of this canonical case; see
  `examples/21_generic_port_subsystems/regen_channel` for the isolated regen MVP.
- See `docs/CANONICAL_MISSION_CASE.md` for the package-level rationale.
