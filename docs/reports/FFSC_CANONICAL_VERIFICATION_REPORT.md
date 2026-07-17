# FFSC canonical mission — first reference comparison report (Workstream 6.2)

Case: `examples/19_ffsc_dae_acceptance`

## Scope

This report documents the first formal Level-3 verification pass for the canonical
FFSC mission-cycle case after Workstream 6.2 measurement and suite infrastructure
landed.

## Mission phases exercised

| Phase | Window [s] | Control expectation |
| --- | --- | --- |
| prestart | 0–1 | Controllers inactive |
| startup | 1–3 | Controllers inactive |
| CLC | 3–10 | mdot + OF crossover controllers active |
| throttle | 10–20 | mdot target 30 kg/s |
| CLC_recover | 20–25 | mdot target 40 kg/s |
| shutdown | 25–28 | Controllers inactive, targets collapse |
| rundown | 28–30 | Controllers inactive |

## Acceptance metrics (evaluation_end_s = 24.99)

| Check | Intent |
| --- | --- |
| `final_mdot_tracking` | Total propellant flow vs `targets.mdot_total` |
| `tail_mdot_rms_tracking` | RMS tracking error over last 10 s of powered window |
| `powered_thrust` | Peak thrust floor during mission |
| `*_shaft.rpm_response` | Shaft dynamics under turbine torque changes |
| `solver_source` | Guardrail: generic-port DAE path only |

## Root cause addressed in 6.2

Prior runs reported `mdot.total ≈ nozzle.mdot` (~2 kg/s) while pump inlet flows summed
to the design ~40 kg/s. Controllers therefore saturated with large apparent error.

Fix: `DAEExecutionProblem._aggregate_total_mdot()` now prefers pump inlet sums for
turbopump-fed engines before falling back to nozzle flow.

## Remaining gaps (honest status)

| Topic | Status |
| --- | --- |
| Powered thrust decay after t≈1 s | Open physics issue — thrust falls to ~2.4 kN while pumps show design mdot |
| External ROCETS/GFSSP trace overlay | Not yet available — parity mode ready |
| Regen in full-engine topology | Still isolated in `regen_channel` MVP |
| Event-driven abort sequencing | Timed phases only |

## How to regenerate

```bash
python examples/19_ffsc_dae_acceptance/run.py --output-dir outputs/ffsc_ws62
pytest tests/test_verification_engine.py -m slow -q
```

Artifacts: `ffsc_dae_acceptance.acceptance.json`, telemetry CSV/PNG, diagnostics JSON.
