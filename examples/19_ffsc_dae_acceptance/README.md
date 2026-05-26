# Example 19: FFSC DAE Acceptance Case

This example is the FFSC DAE acceptance case running through the generic-port
profile solver. It intentionally describes the complete target architecture in
YAML:

- full-flow staged-combustion methalox engine;
- four preburner branch valves;
- two turbopump shafts;
- pump and turbine map references;
- transient valve response definitions;
- operating-condition targets;
- proportional controllers;
- telemetry export channels.

The default `analysis.yaml` executes the decomposed full-port FFSC network with
`solver_source: generic_port`. Reports and console output include the solver
source as a migration guardrail.
It writes:

- `outputs/ffsc_dae_acceptance.csv`
- `outputs/ffsc_dae_acceptance.h5`
- `outputs/ffsc_dae_acceptance.png`
- `outputs/ffsc_dae_acceptance.linearization.json`
- `outputs/ffsc_dae_acceptance.acceptance.json`

The acceptance report validates endpoint target tracking, shaft response,
residual closure, telemetry finiteness, and the generic-port solver-source
guardrail.
