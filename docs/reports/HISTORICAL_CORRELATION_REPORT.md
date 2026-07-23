# Historical correlation suite report (Workstream 6.4)

## Scope

First ATHA external-correlation workflow using seeded analytical and
literature-style references under `verification/historical/`.

## Cases

| Case | Reference | Candidate | Intent |
| --- | --- | --- | --- |
| Pump affinity ramp | `pump_map_affinity_ramp` | example 23 | Map/affinity subsystem credibility |
| Valve orifice step | `valve_orifice_step` | analytical self + orifice oracle | Valve calibration path |
| Chamber startup envelope | `chamber_startup_envelope` | example 25 `chamber_startup_transient` | Startup shape / final Pc-thrust metrics |
| Example 24 parity | same pump oracle CSV | example 23 via `parity.reference_csv` | Retained parity analysis mode |

## Metrics exercised

- RMS / final / peak relative error
- Rise-time and settling-time helpers
- Parity overlays (`parity_delta.csv`)
- Provenance fields from dataset manifests

## Honest limitations

1. Seeded references are synthetic or analytical — not proprietary hot-fire data.
2. Example 25 provides a first-order ignition-envelope rise (not full injector
   network closure); rise-time gates are enabled and pass against the seeded
   envelope with moderately loose tolerances.
3. GFSSP / FullFlow / ROCETS exported traces are not yet checked into the repo.
   The ingestion schema is ready for those packages once exports are available.

## Regeneration

```bash
python scripts/run_historical_correlation.py --output-dir outputs/historical
python -m atha.cli examples/24_pump_map_historical_parity --output-dir outputs/ex24_parity
```
