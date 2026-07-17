# Example 21: Generic-Port Subsystem Matrix

This folder contains Phase E / Workstream 6.2 subsystem examples that run through
`analysis.type: profile` and `DAEExecutionProblem`.

Each subsystem is a small square generic-port balance profile with telemetry,
acceptance, and (where configured) analytical reference checks. These are fast gates
for the generic configuration, balance, output, and verification infrastructure.

## Run the verification suite

```bash
python examples/21_generic_port_subsystems/run_verification_suite.py
```

Or via pytest:

```bash
pytest tests/test_verification_subsystems.py -q
```

See `docs/VERIFICATION_GUIDE.md` for artifact formats and `docs/VERIFICATION_MATRIX.md`
for the component mapping.

Subsystems:

- `valve_pipe_volume`
- `injector_chamber_nozzle`
- `pump_pipe_valve`
- `pump_shaft_turbine`
- `preburner_turbine`
- `chamber_nozzle`
- `regen_channel` (Workstream 6.1 regen MVP)

