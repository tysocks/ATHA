# Missing-physics implementation backlog

Prioritized backlog from Workstream 6.1. Items marked **done in 6.1** were
implemented during this pass. Remaining items are tracked for later workstreams.

## Priority 1 — component derivative completeness

| Item | Status | Notes |
| --- | --- | --- |
| Combustor / preburner / GG `h` derivative | **blocked / documented** | Attempted energy/lag ODEs diverged because `h` is also algebraically owned and synced into `Z`; kept pressure ODE only |
| Separate combustor algebraic vs dynamic `h` ownership | open | Required before a safe dynamic enthalpy ODE can land on the DAE path |
| RegenChannel `T_wall` derivative | **done in 6.1** | `RegenChannelDerivativeContract` + richer residual heat loads |
| GasVolume `P` / `h` derivative + residual | **done in 6.1** | Ideal-gas contracts registered |
| GasGenerator residual lookup key | **done in 6.1** | `residual_contract_for_type("GasGenerator")` |
| Enable pipe dynamics on GG example | **done in 6.1** | Example 20 pipes gained `time_constant` |
| OutletInertia residual + derivative | open | Registered states still DAE-orphaned |
| MetalNode registration | open | Thermal coupling helper exists but is not YAML-usable |

## Priority 2 — rotor / turbomachinery closure

| Item | Status | Notes |
| --- | --- | --- |
| Shaft torque coupling in DAE | existing | Type-based pump/turbine coupling |
| Dual Rotor residual vs coupling model | open | Port builder skips Rotor residual; keep documented until unified |
| Pump NPSH / cavitation limits | open | Not required for current MVP cases |
| Turbine map parity with pump map UX | open | Efficiency / PR map axes weaker than pump φ–ψ path |

## Priority 3 — combustor / GG / preburner fidelity

| Item | Status | Notes |
| --- | --- | --- |
| Simplified FV mass/energy closure | existing / improved | Sufficient for topology + mission control work |
| Low-pressure ignition tables | open | Needed for higher-fidelity start correlation |
| Cantera-backed DAE residual mode | open | OOP Cantera chamber is not the DAE path |
| Choked-nozzle-consistent chamber pressure ODE | **partial in 6.5** | Volume-owned `pressure_residual` soft-out; FFSC sustained thrust still open |

## Priority 4 — regenerative cooling integration

| Item | Status | Notes |
| --- | --- | --- |
| Isolated regen MVP subsystem | **done in 6.1** | `examples/21_generic_port_subsystems/regen_channel` |
| Coupled regen on canonical FFSC | open | Requires coolant branch topology + chamber gas boundary wiring |
| Coolant outlet enthalpy/pressure port residuals | open | Current DAE contract focuses on wall heat balance |
| Discretized multi-node regen chain | open | Later high-fidelity extension |

## Priority 5 — valve / actuator mission logic

| Item | Status | Notes |
| --- | --- | --- |
| Phase-aware controller activation | existing / formalized | `active_phases`, `inactive_phases` |
| Controller reset on phase entry | **done in 6.1** | `reset_on_enter` |
| Hold command when inactive | **done in 6.1** | `hold_when_inactive` |
| Explicit actuator component separate from valve flow | open | Currently transient-block based |
| Event-driven sequence / abort state machine | **done in 6.5 (MVP)** | `advance_when` guards + forced phase ends; full abort FSM still open |

## Priority 6 — full-engine continuity / sustained thrust (Workstream 6.5)

| Item | Status | Notes |
| --- | --- | --- |
| Sustained-thrust acceptance gates | **done in 6.5** | `min_powered_tail_thrust`, `final_thrust_tracking` |
| Combustor volume-owned pressure residual | **done in 6.5** | Soft-out when `volume > 0` |
| FFSC chamber inlet vs pump mdot continuity | open | Peak thrust can pass while powered-tail thrust collapses |
| Preburner → turbine → chamber branch closure | open | Suspected algebraic unlink; tracked by strict tail gate on example 19 |

## Blockers that cannot be fully closed in 6.1 alone

### 1. Full-engine regen coupling

**Problem:** The canonical FFSC example has no coolant regen branch or chamber-wall
thermal ports. Adding regen correctly is a topology + residual + validation change,
not a one-line derivative fix.

**Why not finished here:** Requires redesign of example 19 engine YAML, new thermal
connections, and acceptance criteria for wall temperature / coolant enthalpy rise.
Delivered instead: residual/derivative contracts + isolated regen MVP.

### 2. Event-driven mission sequencer

**Problem:** Real start/shutdown logic often depends on thresholds
(e.g. chamber pressure crossed, shaft speed ready), not only wall-clock phase windows.

**Status after 6.5:** MVP guard-based early phase advance is implemented
(`analysis.time.phases[].advance_when` → `update_forced_phase_ends` /
`resolve_phase_name_with_guards`). Remaining open work: abort handling,
multi-condition guards, and richer sequence-state telemetry.

### 3. Chemistry-accurate DAE combustors

**Problem:** The production DAE path uses simplified finite-volume contracts, while
the richer Cantera OOP chamber is on the legacy path.

**Why not finished here:** Wiring Cantera into every algebraic residual evaluation
is a performance and numerical-stability project of its own. Tracked for later
verification / fidelity workstreams.

### 4. OutletInertia / MetalNode completeness

**Problem:** These types exist in code but are not first-class in the DAE contract
system (`OutletInertia`) or registry (`MetalNode`).

**Why not finished here:** No retained full-engine example depends on them yet.
Priority is lower than pump/shaft/chamber/regen mission-cycle closure.
