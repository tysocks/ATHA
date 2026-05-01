# Component Catalog

Maps ROCKETS modules to ATHA Python components.

## Module Mapping

| ROCKETS Module | ATHA Class | Type | States | Notes |
|---------------|------------|------|--------|-------|
| MCHB01 | CombustionChamber | Dynamic | P, h | H2/O2 or general via Cantera |
| PBRN01 | Preburner | Dynamic | P, h | Fuel-rich or ox-rich |
| NOZL00 | Nozzle | Algebraic | — | Isentropic expansion, JANNAF Cf |
| QCHM01 | ChamberHeatTransfer | Algebraic | — | Bartz correlation |
| QN0Z01 | NozzleHeatTransfer | Algebraic | — | Bartz for nozzle |
| PUMP01 | Pump | Algebraic | — | Constant-density pump with map |
| TURB01 | Turbine | Algebraic | — | Ideal gas turbine with map |
| TURB02 | TurbineSingleFluid | Algebraic | — | Single-constituent turbine |
| ROTR00 | Rotor | Dynamic | ω | Shaft torque balance |
| ROTR01 | RotorWithBreakaway | Dynamic | ω | For start simulation |
| PIPE00 | PipeWithInertia | Dynamic | ṁ | Inertia + friction loss |
| PIPE01 | PipeAlgebraic | Algebraic | — | Quasi-steady incompressible |
| PIPE02 | OrificeCompressible | Algebraic | — | Choked/subsonic compressible flow |
| PIPE03 | PipeWithInertiaGravity | Dynamic | ṁ | Adds elevation change |
| VALV00 | Valve | Algebraic | — | Variable-area incompressible |
| VOLM00/01 | Volume | Dynamic | P, h | Multi-flow lumped volume |
| METL00 | MetalNode | Dynamic | T_wall | Lumped thermal mass |

## Port Interface Summary

### FluidPort
- ṁ [kg/s]: mass flow rate (positive = flow into component)
- P [Pa]: pressure at port face
- h [J/kg]: specific enthalpy

### ShaftPort
- ω [rad/s]: rotational speed
- τ [N⋅m]: torque (positive = turbine delivering TO shaft)

### ThermalPort
- T_wall [K]: wall temperature
- Q̇ [W]: heat flux (positive = into metal node)

## Component Parameter Reference

### CombustionChamber
| Parameter | Symbol | Units | Typical |
|-----------|--------|-------|---------|
| Volume | V_ch | m³ | 0.01–0.1 |
| Characteristic length | L* | m | 0.7–1.0 (H2/O2) |
| Combustion efficiency | η_c* | — | 0.97–0.99 |

### Pump
| Parameter | Symbol | Units | Notes |
|-----------|--------|-------|-------|
| Impeller diameter | D | m | Sets map scale |
| Pump map file | — | CSV | phi, psi, eta_p columns |

### Rotor
| Parameter | Symbol | Units | Notes |
|-----------|--------|-------|-------|
| Moment of inertia | I | kg⋅m² | Polar moment |
| Friction coefficient | k_f | N⋅m⋅s/rad | Bearing friction |

### Nozzle
| Parameter | Symbol | Units | Notes |
|-----------|--------|-------|-------|
| Throat area | A_t | m² | Sized from thrust/Pc |
| Exit area | A_e | m² | A_t × ε |
| Divergence half-angle | α | degrees | 15° typical conical |

## Example Engine Topologies

### Gas Generator Cycle
```
LOX tank → valve → pump → injector → main chamber → nozzle
LH2 tank → valve → pump → GG injector → gas generator → turbine → exhaust
                   pump ← turbine (shaft)
```

### Staged Combustion (Full-Flow)
```
LOX → pump → ox preburner → ox turbine → main injector → chamber → nozzle
LH2 → pump → fuel preburner → fuel turbine → main injector
            ↑                              ↑
        (shaft from ox turb)         (shaft from fuel turb)
```
