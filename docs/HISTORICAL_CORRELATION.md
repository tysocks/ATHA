# Historical / external correlation workflow (Workstream 6.4)

This guide describes how ATHA ingests external or literature reference data and
compares it to ATHA telemetry using physically meaningful metrics.

## Goals

- Make ATHA credible against maps, analytical oracles, and literature-style traces
- Keep proprietary hot-fire ingestion path ready even when only synthetic seeds exist
- Emit correlation reports with provenance, overlays, and pass/fail thresholds

## Reference dataset package layout

Each dataset folder under `verification/historical/` contains:

```text
verification/historical/<dataset_id>/
  manifest.yaml
  trace.csv          # or other path listed in data_file
```

### Required manifest fields

| Field | Meaning |
| --- | --- |
| `id` | Stable dataset identifier |
| `title` | Human-readable name |
| `source` | Origin label (map CSV, literature, GFSSP export, …) |
| `provenance` | How the data was created / digitized / filtered |
| `allowed_use` | e.g. `verification_only` |
| `category` | `analytical`, `literature_synthetic`, `hotfire`, `external_export` |
| `data_file` | Relative path to the trace |
| `time_alignment` | Time column, offset/scale, optional trim window |
| `channels` | Mapping from reference columns to ATHA aliases |

Example channel entry:

```yaml
channels:
  - reference_channel: MDOT
    atha_channel: MDOT
    units: kg/s
    scale: 1.0
    offset: 0.0
```

Load in Python:

```python
from atha.validation.reference_data import load_reference_dataset, discover_reference_datasets

dataset = load_reference_dataset("verification/historical/valve_orifice_step")
time, channels = dataset.load_series()
```

## Physical metrics

`atha.validation.historical_correlation` evaluates:

| Metric | Meaning |
| --- | --- |
| `rms_rel` | Relative RMS trace error |
| `final_rel` | Relative final steady-state error |
| `peak_rel` | Relative peak-value error |
| `rise_time_error_s` | 10–90% rise-time difference |
| `settling_time_error_s` | Settling-time difference inside a band |
| `integrated_rel` | Relative integrated error (impulse / mass-flow style) |
| `overshoot` | Diagnostic overshoot vs final value |

Parity metrics (`max_abs_error`, `rms_error`, `final_abs_error`) can be attached in
the same report when `parity_channels` are provided.

## Seeded datasets

| Dataset | Purpose |
| --- | --- |
| `pump_map_affinity_ramp` | Affinity-law pump ramp vs example 23 |
| `valve_orifice_step` | Textbook orifice opening characterization |
| `chamber_startup_envelope` | Literature-style Pc/thrust startup envelope |

These seeds are **not** proprietary hot-fire recordings. They establish the
ingestion and reporting workflow until real test data or external-package
exports are attached.

## Run the correlation suite

```bash
python scripts/run_historical_correlation.py
python scripts/run_historical_correlation.py --list-datasets
```

Outputs land under `outputs/historical/` and include:

- `*.correlation.json` — metric summaries + provenance
- `*.parity.json` / `*.parity_delta.csv` — overlay artifacts when configured
- `historical_suite_report.json` — batch summary

## Retained parity example

Example 24 compares example 23 against the pump affinity oracle CSV using
`analysis.type: parity` with `parity.reference_csv`:

```bash
python -m atha.cli examples/24_pump_map_historical_parity --progress
```

## Adding a real hot-fire or GFSSP/FullFlow export

1. Create `verification/historical/<id>/manifest.yaml` with provenance and units.
2. Drop the exported CSV/HDF5 next to the manifest and set `data_file`.
3. Map channels to ATHA telemetry aliases.
4. Choose time alignment / trim windows for ignition or valve-open sync.
5. Add a case to `scripts/run_historical_correlation.py` (or a dedicated config).
6. Document assumptions in `docs/reports/`.

## Related docs

- `docs/VERIFICATION_GUIDE.md` — acceptance / regression / suite gates
- `docs/VERIFICATION_MATRIX.md` — component/case status
- `docs/reports/FFSC_CANONICAL_VERIFICATION_REPORT.md` — Level-3 engine report
- `CONTRIBUTING.md` — how to add verification cases
