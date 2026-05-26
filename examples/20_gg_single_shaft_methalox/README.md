# Example 20: Methalox Single-Shaft Gas Generator

The default `analysis.yaml` is a generic-port DAE run for the methalox
single-shaft gas-generator cycle. Historical alternate configs have been
removed from the retained example.

Run it with:

```powershell
.venv\Scripts\python.exe examples\20_gg_single_shaft_methalox\run.py
```

The cycle is YAML-defined:

- LOX and methane pumps on one shaft.
- Pump discharge pipes feeding flow splitters.
- Main branches through pipes, main valves, pipes, injectors, chamber, and
  nozzle.
- Generator branches through pipes, generator valves, pipes, injectors,
  generator, turbine, exhaust pipe, and ambient.
- Two proportional controllers evaluated at `2 Hz`:
  - methane generator valve controls total mass flow;
  - LOX generator valve controls mixture ratio.
- Main propellant valves close at `25 s` through `timings.yaml`.

Outputs are written to `outputs/gg_single_shaft_methalox.*`. Console output,
provenance, and acceptance reports identify the default run as
`solver_source: generic_port`.
