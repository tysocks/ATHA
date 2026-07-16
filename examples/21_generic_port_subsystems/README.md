# Example 21: Generic-Port Subsystem Matrix

This folder contains Phase E subsystem examples that run through
`analysis.type: profile` and `DAEExecutionProblem`.

Each subsystem is a small square generic-port balance profile with telemetry and
an acceptance report. These are intentionally fast gates for the generic
configuration, balance, output, and acceptance infrastructure. The retained
example 19 and 20 detailed engine files remain the source for later physical
component-by-component closure work.

Subsystems:

- `valve_pipe_volume`
- `injector_chamber_nozzle`
- `pump_pipe_valve`
- `pump_shaft_turbine`
- `preburner_turbine`
- `chamber_nozzle`
- `regen_channel` (Workstream 6.1 regen MVP)

