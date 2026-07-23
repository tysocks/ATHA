# ATHA component maturity matrix

Audit date: Workstream 6.1 implementation pass.

Legend:

- **Residual**: algebraic residual contract present for generic-port DAE path
- **Derivative**: plant ODE contract present for DAE state integration
- **Mission relevance**: importance to start / CLC / throttle / shutdown modeling
- **Verification**: current example / acceptance coverage

| Component | Residual | Derivative | States | Mission relevance | Example coverage | Maturity notes |
| --- | --- | --- | --- | --- | --- | --- |
| BoundarySource | yes | n/a | none | high | 19, 20, 22, 23, 21 | Stable anchors |
| BoundarySink | yes | n/a | none | high | 19, 20, 22, 23, 21 | Stable anchors |
| Valve | yes | no (actuator via transients) | none | high | 19, 20, 22, 23, 21 | Flow residual mature; actuator dynamics via YAML transients |
| Pipe | yes | yes | `mdot` | high | 19 (τ), 20 (τ added), 21, 22 | Needs `time_constant` or `inertance` for nonzero ODE |
| MassFlowInjector | yes | n/a | none | high | 19, 20, 22, 21 | Algebraic ΔP law |
| FlowSplitter | yes | n/a | none | high | 19, 20, 22, 21 | Fixed or hydraulic split |
| CombustionChamber | yes | yes (`P`; `h` algebraic) | `P`, `h` | high | 19, 20, 22, 21 | Simplified FV; dynamic `h` deferred due to dual ownership |
| Preburner | yes | yes (`P`; `h` algebraic) | `P`, `h` | high | 19, 20 | Same FV combustor contract |
| GasGenerator | yes | yes (`P`; `h` algebraic) | `P`, `h` | high | 22 | Own residual key now registered |
| Nozzle | yes | n/a | none | high | 19, 20, 22, 21 | Algebraic choked / thrust closure |
| Pump | yes | n/a | none | high | 19, 20, 22, 23, 21 | Map / affinity supported; algebraic |
| Turbine | yes | n/a | none | high | 19, 20, 22, 21 | Algebraic power / map support |
| Rotor | yes* | yes (`omega`) | `omega` | high | 19, 20, 22, 21 | Residual skipped in port builder; shaft coupling + ODE used |
| RegenChannel | yes | yes (`T_wall`) | `T_wall` | medium→high | 21 regen MVP | Coupled full-engine regen still pending |
| GasVolume / Volume | yes | yes (`P`, `h`) | `P`, `h` | medium | 21 valve_pipe_volume | Ideal-gas DAE contracts added in 6.1 |
| OutletInertia / Outlet | no | no | `mdot` | low/medium | limited | Still DAE-orphaned |
| OrificeCompressible | yes | n/a | none | medium | limited | Shares valve flow contract |
| MetalNode | no | no | n/a | medium (thermal) | none | Unregistered; not YAML-usable |

\* Rotor residual contract exists but `PortNetworkBuilder` intentionally excludes it and uses shaft coupling residuals instead.

## Strongest current areas

- Port-network assembly and DAE orchestration
- Phase-aware controllers and schedules
- Pump / turbine / shaft topology for GG and FFSC examples
- Example-driven acceptance / regression reporting

## Weakest current areas

- OutletInertia DAE closure
- Full-engine regen coupling into chamber/nozzle walls
- Chemistry-accurate combustor thermochemistry on the DAE path
- Event-driven sequence state machines beyond timed phases
- Systematic component MVP verification against external tools

## Recommended next maturity upgrades

1. OutletInertia residual + derivative contracts
2. Coupled regen on the canonical FFSC or GG engine
3. Component-level MVP verification cases for every high-relevance row
4. Optional Cantera-backed combustor residual mode
5. Register or remove `MetalNode`
