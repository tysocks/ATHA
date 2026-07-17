# ATHA verification matrix

Workstream 6.2 formalizes a four-level verification ladder. This matrix maps each
major component and retained example to its gate, oracle, and current status.

## Verification levels

| Level | Name | Intent |
| --- | --- | --- |
| 0 | Unit / math | Analytical identities, bounds, derivative/residual sign sanity |
| 1 | Component MVP | Isolated or near-isolated component behavior |
| 2 | Subsystem reference | Coupled chains with fast acceptance + reference checks |
| 3 | Full engine mission | Startup, throttle, shutdown, controller transitions |

## Component matrix

| Component | Level 1 / 2 case | Oracle / reference | Automated gate | Status |
| --- | --- | --- | --- | --- |
| Valve | `valve_pipe_volume` | Future orifice table (`verification/references/valve_orifice_design_point.csv`) | acceptance + pytest | **Pass** (balance gate) |
| Pipe / volume | `valve_pipe_volume` | Lumped pressure / mdot balance | acceptance + pytest | **Pass** |
| Pump | `pump_pipe_valve`, `lox_pump_map` | Map table + affinity laws (`reference_checks`) | acceptance/regression + pytest | **Pass** |
| Turbine | `preburner_turbine`, `pump_shaft_turbine` | Energy balance / shaft torque closure | acceptance + pytest | **Pass** |
| Rotor / shaft | `pump_shaft_turbine` | Inertia response, rpm delta | acceptance + pytest | **Pass** |
| Injector | `injector_chamber_nozzle` | ΔP–mdot balance stub | acceptance + pytest | **Pass** |
| Chamber / preburner / GG | `injector_chamber_nozzle`, `preburner_turbine` | P/T/OF balance stubs | acceptance + pytest | **Pass** |
| Nozzle | `chamber_nozzle` | Thrust = 3500×mdot balance law + c* target | acceptance + reference + pytest | **Pass** |
| Regen channel | `regen_channel` | Wall heating sign, Q_hot > Q_cool | acceptance + reference + pytest | **Pass** |
| Full FFSC engine | `19_ffsc_dae_acceptance` | Mission schedule + controller phases (`docs/CANONICAL_MISSION_CASE.md`) | acceptance + pytest (`slow`) | **In progress** (mdot tracking fix landed in 6.2) |
| GG single-shaft | `20_gg_single_shaft_methalox` | Mission profile + regression windows | acceptance/regression | **Pass** (existing gates) |

## Subsystem suite (`examples/21_generic_port_subsystems/`)

| Folder | Verification purpose | Acceptance tolerances |
| --- | --- | --- |
| `valve_pipe_volume` | Valve + pipe + volume pressure ramp | `final_mdot_rel ≤ 0.05` |
| `pump_pipe_valve` | Pump ΔP/power with pipe and valve | `final_mdot_rel ≤ 0.05` |
| `pump_shaft_turbine` | Shaft speed response | `min_shaft_speed_delta_rpm ≥ 1000` |
| `injector_chamber_nozzle` | Injector → chamber → nozzle | `min_peak_thrust ≥ 10 kN` |
| `chamber_nozzle` | Chamber/nozzle thrust closure | `min_peak_thrust ≥ 10 kN` + reference law |
| `preburner_turbine` | Preburner branch + turbine | standard mdot/OF |
| `regen_channel` | Regen MVP thermal ODE | required thermal paths + reference sign checks |

Run the full fast suite:

```bash
python examples/21_generic_port_subsystems/run_verification_suite.py
pytest tests/test_verification_subsystems.py
```

## External comparison targets (6.2D roadmap)

| ATHA case | External benchmark | Artifact location |
| --- | --- | --- |
| Pump map (example 23) | Map CSV + affinity laws | `verification/references/pump_map_design_point.csv` |
| Valve orifice | Textbook incompressible flow | `verification/references/valve_orifice_design_point.csv` |
| Nozzle thrust | Cf·At·(Pc−Pa) | `verification/references/nozzle_thrust_design_point.csv` |
| FFSC mission | ROCETS-style phase semantics | `docs/CANONICAL_MISSION_CASE.md` |
| Pump-shaft-turbine | GFSSP / FullFlow-style shaft transients | future exported CSV references |
| Injector-chamber-nozzle | JANNAF / ESPSS reduced cases | future literature traces |

## Code entry points

| Module | Role |
| --- | --- |
| `atha/validation/acceptance.py` | Generic-port acceptance JSON reports |
| `atha/validation/reference_checks.py` | Analytical oracles and CSV comparison |
| `atha/validation/verification_suite.py` | Case registry + batch runner |
| `atha/validation/parity.py` | Reference vs candidate parity mode |
| `tests/test_level0_reference_checks.py` | Level 0 unit gates |
| `tests/test_verification_subsystems.py` | Level 2 automated gates |
| `tests/test_verification_engine.py` | Level 3 slow engine gate |
