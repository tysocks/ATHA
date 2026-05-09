# ATHA Validation Matrix

This matrix tracks the ROCETS-style validation ladder for ATHA. The intent is
to move from isolated component behavior to closed-loop full-engine acceptance,
with every tier producing machine-readable artifacts.

| Tier | Scope | Current Coverage | Artifact |
| --- | --- | --- | --- |
| Component verification | Single component residuals, maps, transients | Unit tests for maps, transient blocks, valve/nozzle residual contracts | Pytest results |
| Subsystem verification | Valve-volume, two-valve chain, pressure-fed TCA | Examples 15-18 execute and export telemetry | CSV, HDF5, plots |
| Full engine steady trim | Connected full-cycle trim | Reduced FFSC endpoint closure in example 19; true port trim pending | Acceptance JSON |
| Throttle transient | Target changes and controller response | Example 19 mdot target step-down/step-up checks | Acceptance JSON |
| Start transient | Closed-to-open or dwell-to-run start profile | Examples 14-17 cover valve starts; no full-cycle start acceptance yet | CSV, plots |
| Shutdown transient | Controlled shutdown profile | Not implemented | Pending |
| Closed-loop controller transient | Controller commands drive component states | Example 18 and example 19 controller tracking checks | CSV, acceptance JSON |
| Linearization | A/B/C/D around operating point | Example 19 reduced linearization artifact; pressure-fed trim linearization | Linearization JSON |

## Phase 17 Acceptance Artifacts

Example 19 writes:

- `outputs/ffsc_dae_acceptance.acceptance.json`
- `outputs/ffsc_dae_acceptance.linearization.json`
- `outputs/ffsc_dae_acceptance.csv`
- `outputs/ffsc_dae_acceptance.h5`
- `outputs/ffsc_dae_acceptance.png`

The current example 19 acceptance report is a reduced-order FFSC gate. It is
not the final arbitrary port-variable validation. The report categories are:

- `numerical`
- `physical_model`
- `controller`
- `telemetry`
- `linearization`

## Remaining Full-Port Validation Requirements

The full ROCETS-like validation gate requires Phase 20 automatic port DAE
assembly:

- per-port pressure, mass-flow, enthalpy/temperature, and density histories;
- connection continuity residuals;
- pipe inertia residuals;
- preburner and chamber mass/energy residuals;
- pump and turbine map residuals;
- true steady trim closure before transient integration.
