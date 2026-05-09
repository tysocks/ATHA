# Example 19: FFSC DAE Acceptance Case

This example is the solved reduced-order FFSC DAE acceptance case and the
configuration target for the remaining full arbitrary port solve work. It
intentionally describes the complete target architecture in YAML:

- full-flow staged-combustion methalox engine;
- four preburner branch valves;
- two turbopump shafts;
- pump and turbine map references;
- transient valve response definitions;
- operating-condition targets;
- proportional controllers;
- telemetry export channels.

The current ATHA runner executes this case with a reduced-order FFSC DAE model.
It writes:

- `outputs/ffsc_dae_acceptance.csv`
- `outputs/ffsc_dae_acceptance.h5`
- `outputs/ffsc_dae_acceptance.png`
- `outputs/ffsc_dae_acceptance.linearization.json`
- `outputs/ffsc_dae_acceptance.acceptance.json`

The acceptance report validates reduced-model endpoint closure, target
tracking, shaft response, residual closure, telemetry finiteness, and
linearization artifact generation.

What is still pending:

- automatic fluid-port unknown generation;
- connection pressure/enthalpy/mass residuals;
- pipe inertia residual contracts;
- chamber/preburner thermochemical residuals;
- map-backed pump/turbine residuals;
- true full-port steady trim before transient integration.

Those items are tracked as the full arbitrary ROCETS-like port solve in
`development/2026-05-08-general-config-runner-architecture.md`.
