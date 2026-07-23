# Canonical full-engine mission case

## Selection

**Canonical case:** `examples/19_ffsc_dae_acceptance`

This is the Workstream 6.1 integration target for end-to-end ATHA development.

## Why example 19

| Criterion | Example 19 FFSC | Example 20 GG | Example 22 two-shaft GG |
| --- | --- | --- | --- |
| Cycle completeness | Dual preburner FFSC | Single-shaft GG | Two-shaft GG |
| Maps | Pump + turbine | Pump + turbine | None |
| Pipe dynamics | `time_constant` present | Added in 6.1 | Limited |
| Controllers | Phase-aware P | PID | PID |
| Acceptance depth | Acceptance + linearization + expected behavior | Acceptance + regression | Acceptance + regression |
| Mission schedule richness | Startup / CLC / throttle / recover / shutdown / rundown | Startup / CLC / shutdown | Strong target rundown |

Example 19 best stresses the retained package architecture. Examples 20 and 22 remain important GG templates.

## Mission phases in the canonical case

| Phase | Time window | Intent |
| --- | --- | --- |
| `prestart` | 0–1 s | Low open-loop target, controllers inactive |
| `startup` | 1–3 s | Open-loop ramp toward mainstage |
| `CLC` | 3–10 s | Closed-loop regulation at design |
| `throttle` | 10–20 s | Closed-loop throttle to 30 kg/s |
| `CLC_recover` | 20–25 s | Recover to design before shutdown |
| `shutdown` | 25–28 s | Valve close / target collapse |
| `rundown` | 28–30 s | Final coast-down |

## Control scheme by phase

- Controllers active in: `CLC`, `throttle`, `CLC_recover`
- Controllers inactive in: `prestart`, `startup`, `shutdown`, `rundown`
- `reset_on_enter: [CLC]` clears controller memory on closed-loop entry
- `hold_when_inactive: true` freezes last command outside active phases
- Shutdown valve motion remains timing-driven

## Required hardware coverage for this case

Present today:

- chamber, injectors, nozzle
- valves, pipes, splitters
- pumps, turbines, rotors
- preburners
- boundaries

Not yet in the canonical case:

- regenerative cooling coupled into the FFSC topology
- chilldown physics beyond a named `prestart` window
- event-driven aborts / ignition detection state machine

## Companion cases

| Case | Role |
| --- | --- |
| `examples/20_gg_single_shaft_methalox` | GG + PID + regression template |
| `examples/22_ethanol_lox_5kn_two_shaft_gg` | Mission-profile / rundown schedule template |
| `examples/23_single_lox_pump_map` | Pump-map verification seed |
| `examples/21_generic_port_subsystems/regen_channel` | Regen MVP subsystem |

## Definition of done for this canonical case

The case is considered mission-cycle ready for near-term ATHA when it:

1. Runs prestart → startup → CLC → throttle → recover → shutdown → rundown without ad hoc Python.
2. Activates and deactivates controllers by phase.
3. Integrates nonzero plant derivatives for shafts, pipes, and combustor states.
4. Passes acceptance with documented tolerances.
5. Has an accompanying maturity/backlog note for remaining physics gaps.
