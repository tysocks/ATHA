# Architecture Decisions

---

## Decision 1: Compile Step (Engineering Model → Numerical Model)

**What:** `Engine.compile()` converts the component object graph into `EngineLayout`,
a flat numerical representation with pre-allocated arrays and a fixed evaluation order.

**Why:** Without compilation, every ODE function call involves Python dictionary
lookups, dynamic dispatch, and attribute access across dozens of component objects.
This makes the simulation 10–100x slower than necessary and incompatible with
scipy's Radau solver which calls the RHS thousands of times per second.

After compilation, `EngineLayout.evaluate(t, X, Z)` operates on numpy arrays
in a fixed sequence, enabling pre-allocated workspace arrays and sparse Jacobian exploitation.

**How to apply:** Never call component methods directly during integration.
Always go through `EngineLayout.evaluate()`.

---

## Decision 2: P and h as Volume States

**What:** Lumped volumes use pressure P [Pa] and specific enthalpy h [J/kg] as
integration states, not density ρ and internal energy u.

**Why:** The ROCKETS system identified this exact issue: ρ and u produce numerically
ill-conditioned Newton iterations in liquid propellant systems because:
- Pressure is extremely sensitive to density for liquids (bulk modulus ~GPa)
- Initial guesses for u are hard to specify physically

CoolProp's `state_from_Ph(P, h)` is the primary hot-path function for this reason.

---

## Decision 3: DAE-to-ODE via Inner Newton

**What:** The engine system is a DAE system. ATHA converts it to a pure ODE by
eliminating algebraic variables Z at each integration step via an inner Newton iteration.

**Why:** SciPy's `solve_ivp` with Radau handles stiff ODEs extremely well but
is not a DAE solver. The DAE-to-ODE conversion works here because the algebraic
constraints are index-1: Z appears explicitly in the residuals without differentiation.

**How to apply:** `TransientSolver._ode_rhs(t, X)` calls `inner_newton_solve(t, X, Z_guess)`
to find Z, then returns `dX/dt = layout.evaluate(t, X, Z)[0]`.
Z_guess is updated in-place to warm-start the next step.

---

## Decision 4: Cantera Equilibrium Caching

**What:** Pre-compute (MR, P) equilibrium grid at component initialization,
fit 2D spline, interpolate during integration.

**Why:** `Cantera.equilibrate('HP')` is 0.5–2ms per call. At ~1000 RHS
evaluations/second during transients, raw Cantera calls are prohibitive.

**How to apply:**
- Default grid: 20 × 20 points over MR ∈ [0.5×MR_design, 2.0×MR_design]
  and P ∈ [0.3×P_design, 1.2×P_design]
- Spline built by `scipy.interpolate.RectBivariateSpline`
- Grid rebuilds automatically when operating range is exceeded

---

## Decision 5: Sparse Jacobian with Graph Coloring

**What:** The Newton solver uses the engine's connectivity graph to build a
sparse Jacobian pattern, then uses graph coloring to group non-overlapping
columns and evaluate multiple perturbations simultaneously.

**Why:** A 122-state TTBE engine has a 122×122 Jacobian. Without sparsity,
this requires 122 function evaluations per Newton step. The physical connectivity
means ~90% of entries are zero. Graph coloring reduces evaluations to ~10–20.

---

## Decision 6: Three Port Domains

**What:** FluidPort (ṁ, P, h), ShaftPort (ω, τ), ThermalPort (T_wall, Q̇).
Connections only allowed between matching domains.

**Why:** Enforcing domain at connection time catches wiring errors early.
The three domains correspond exactly to the physical energy transfer mechanisms.

---

## Decision 7: SI Units Throughout

**What:** All internal calculations use SI: Pa, K, J/kg, kg/s, rad/s, N, m, m².
Unit conversion only at user-facing API.

**Why:** Mixed-unit bugs are a common source of simulation errors. Enforcing SI
internally eliminates an entire class of bugs.

---

## Decision 8: Multi-Engine Cycle Support via Topology

**What:** Engine cycles are defined purely by component topology, not by
cycle-specific code paths.

**Gas generator:** split fraction of propellant → small combustor → turbine → exhaust
**Staged combustion:** preburner → turbine → main chamber (all propellant)
**Expander:** regenerative cooling channel heats fuel → drives turbine
**Each is just a different graph of the same component types.**

---

## Pitfalls

1. **CoolProp phase transitions:** Near saturation, properties are discontinuous.
   Volume components must detect two-phase and handle explicitly.

2. **Choked nozzle Jacobian singularity:** At exactly choked, the subsonic flow
   formula has a zero denominator. Use hyperbolic blend to maintain smooth Jacobian.

3. **Zero-speed pump/turbine init:** Performance maps undefined at ω=0. During
   engine start, replace map-based equations with orifice models until ω > ω_min.

4. **Circular algebraic dependencies:** Staged combustion engines have inherent
   circular flows. The global Newton solver on the full algebraic vector handles them.

5. **Mixture ratio bounds:** Add event functions to halt integration when MR
   leaves [0.1, 100].

6. **Cantera mechanism consistency:** Use the same mechanism file for all components.
