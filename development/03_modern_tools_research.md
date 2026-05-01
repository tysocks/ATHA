# Modern Python Tools for Rocket Engine Simulation

---

## NASA CEA (Chemical Equilibrium with Applications)

- Gold standard since 1940s, continuously refined at NASA Glenn
- Calculates equilibrium compositions via free-energy minimization
- 1,900+ species database (gas + condensed)
- Outputs: T_ad, γ, c*, Isp, Cf, molecular weight, composition
- Supports: frozen and shifting equilibrium, Chapman-Jouguet detonations
- Versions: CEA2 (2002, legacy FORTRAN) and CEA2022 (open source)
- **ATHA approach:** Use Cantera instead — same physics, Python-native, faster to integrate

## Cantera

- Open-source C++/Python library for chemical kinetics, thermodynamics, transport
- Equilibrium via Gibbs free energy minimization: `gas.equilibrate('HP')`
- Detailed reaction mechanisms (thousands of species/reactions)
- Transport properties: diffusion, thermal conductivity, viscosity
- NASA 7-coefficient polynomial thermodynamics (same format as JANNAF/CEA)
- **Key limitation:** `equilibrate('HP')` is 0.5–2ms per call → must cache results
  Pre-compute (MR, P) grid → 2D spline → interpolate during integration

## CoolProp

- Open-source C++/Python thermophysical property library
- State-of-the-art equations of state for real fluids
- Primary call: `AS.update(CP.HmassP_INPUTS, h, P)` → FluidState (P, h hot path)
- Tabular backends (TTSE, BICUBIC) give 10–100x speedup after setup
- Fluids relevant to rocketry:
  - LOX: 'Oxygen' (Tc=154.6K, Pc=50.4bar)
  - LH2: 'Hydrogen' (Tc=33.1K, Pc=13.0bar)
  - LCH4: 'Methane' (Tc=190.6K, Pc=46.1bar)
  - RP-1: 'n-Dodecane' (approximation) — no official RP-1 fluid
- **Critical pitfall:** Phase boundaries → discontinuous properties → must detect
  two-phase states and handle explicitly. Near critical point properties diverge.

## SciPy ODE Solvers

| Solver | Method | Best For |
|--------|--------|---------|
| RK45 | Explicit RK | Non-stiff ODEs |
| Radau | Implicit RK | **Stiff ODEs ← primary choice for ATHA** |
| BDF | Adams/BDF | Stiff ODEs (alternative) |
| LSODA | Auto-switch | Unknown stiffness |

Rocket engine transients are inherently stiff due to:
- Fast pressure wave propagation (acoustic timescale ~ms) vs.
  slow thermal dynamics (thermal timescale ~seconds)
- SciPy Radau is the correct choice for ATHA transient integration

**Sparse Jacobian:** `solve_ivp(jac_sparsity=sparsity_matrix)` enables
sparse finite-difference Jacobian computation → critical for large engines (100+ states)

## RPA (Rocket Propulsion Analysis)

- Java-based commercial/academic tool
- Gibbs free energy minimization (same as Cantera)
- Bartz and Ievlev heat transfer methods
- Thermal analysis: regenerative cooling, film cooling, radiation
- Off-design cycle analysis with turbomachinery
- **Use for:** Validation of ATHA performance predictions

## NPSS (Numerical Propulsion System Simulation)

- NASA Glenn → now Southwest Research Institute
- Object-oriented, non-linear thermodynamic modeling
- Multi-fidelity "zooming" (0D coupled to 3D CFD)
- CORBA middleware for distributed simulation
- **Inspiration:** ATHA's Engine.compile() pattern mirrors NPSS's compilation philosophy

## ESPSS (European Space Propulsion System Simulation)

- ESA toolkit on EcosimPro platform
- Object-oriented, graphical and scripted interfaces
- Transient capability for full engine cycles
- **Reference:** Confirms the modular ODE/DAE approach for European engines

## Key Python Libraries Summary

```python
# Core numerical
numpy           # array math, linear algebra
scipy           # ODE solvers, optimization, sparse matrices
                # scipy.integrate.solve_ivp (Radau for stiff)
                # scipy.optimize.fsolve / root (Newton-Raphson)
                # scipy.sparse (Jacobian sparsity)

# Thermodynamics  
cantera         # combustion chemistry, equilibrium, transport
CoolProp        # fluid thermophysical properties (LOX, LH2, LCH4)

# Utilities
pydantic        # config validation / schema
matplotlib      # analysis plots
plotly          # interactive exploration plots

# Optional acceleration
numba           # JIT compilation for hot loops
```
