# Verification reference traces

Analytical and exported reference data used by Workstream 6.2 comparison harnesses.

| File | Purpose |
| --- | --- |
| `valve_orifice_design_point.csv` | Incompressible orifice mdot at one design ΔP |
| `nozzle_thrust_design_point.csv` | Thrust coefficient × area × ΔP oracle |
| `pump_map_design_point.csv` | Example 23 map efficiency at a tabulated (φ, ψ) point |

These files are consumed by `atha.validation.reference_checks` and the verification
suite post-processors. They are intentionally small single-point tables so the
gates stay fast and deterministic.
