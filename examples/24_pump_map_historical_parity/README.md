# Example 24: Pump-map historical / literature parity

Compares the retained example-23 LOx pump-map transient against the synthetic
affinity-law reference package under
`verification/historical/pump_map_affinity_ramp/`.

## Run

```bash
python -m atha.cli examples/24_pump_map_historical_parity --progress
```

Artifacts:

- `pump_map_historical_parity.parity.json`
- `pump_map_historical_parity.parity_delta.csv`
- candidate telemetry under `candidate/`

This is the first retained Workstream 6.4 parity example that uses an external
CSV oracle (`analysis.parity.reference_csv`) instead of a second ATHA config.
