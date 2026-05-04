# ATHA Simulation Technical Reference

Complete reference for ATHA internals: solver algorithms, physics models, simulation setup, and result extraction.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [State Vector and Compile Step](#2-state-vector-and-compile-step)
3. [Physics Models — Dynamic States](#3-physics-models--dynamic-states)
4. [Physics Models — Algebraic Components](#4-physics-models--algebraic-components)
5. [Thermodynamic Backends](#5-thermodynamic-backends)
6. [Port System and Component Wiring](#6-port-system-and-component-wiring)
7. [Steady-State Solver](#7-steady-state-solver)
8. [Transient Solver](#8-transient-solver)
9. [JANNAF Performance Module](#9-jannaf-performance-module)
10. [Setting Up a Simulation — Step-by-Step](#10-setting-up-a-simulation--step-by-step)
11. [Extracting and Interpreting Results](#11-extracting-and-interpreting-results)
12. [Component Reference](#12-component-reference)
13. [Boundary Conditions Dictionary Patterns](#13-boundary-conditions-dictionary-patterns)
14. [Numerical Considerations and Failure Modes](#14-numerical-considerations-and-failure-modes)

---

## 1. System Architecture

ATHA is organized as a three-layer stack:

```
User code
    │
    ▼
Engine / EngineLayout  (atha/core/engine.py)
    │   compile() converts Python objects → flat numpy arrays
    ▼
Solver  (atha/solver/)
    │   operates on EngineLayout + state vector X only
    ▼
Components  (atha/components/)
    │   compute_outputs(), get_state_derivatives(), get_residuals()
    ▼
ThermoBackend  (atha/thermo/)
        state_from_Ph(), state_from_PT(), isentropic_expansion()
```

### Design rationale

Python object-attribute access and dictionary lookups are expensive when called thousands of times per second inside a SciPy ODE integrator. The compile step converts all component-level data into flat numpy arrays with pre-assigned integer indices. After compilation, the solver hot path consists only of numpy operations and the thin `rhs()` function — no Python-level dictionary building except at boundary condition injection.

This mirrors the ROCKETS system's "Configuration Processor" which generated a FORTRAN main program at compile time, eliminating dynamic dispatch during integration.

---

## 2. State Vector and Compile Step

### State vector X

`X` is a 1-D `numpy.ndarray` of shape `(n_states,)` in SI units throughout (Pa, J/kg, rad/s, kg/s, K). Each dynamic component owns a contiguous slice of X.

### EngineLayout

`Engine.compile()` returns an `EngineLayout` object:

```python
@dataclass
class EngineLayout:
    components:    List[BaseComponent]   # evaluation order
    state_offsets: Dict[str, int]        # comp.name → start index in X
    alg_offsets:   Dict[str, int]        # comp.name → start index in Z
    n_states:      int
    n_algebraic:   int
```

**`state_offsets`** maps each component name to where its states begin in X. If a component has states `["P", "h"]` and `state_offsets["chamber"] == 4`, then `X[4]` is `chamber.P` and `X[5]` is `chamber.h`. Components are processed in insertion order.

### assemble_state_vector()

Reads `comp._state_values` (a dict populated at `__init__` time from constructor arguments) and packs them into a new numpy array:

```python
def assemble_state_vector(self) -> np.ndarray:
    X = np.zeros(self.n_states)
    for comp in self.components:
        off = self.state_offsets.get(comp.name)
        if off is not None:
            for i, name in enumerate(comp.state_names):
                X[off + i] = comp._state_values[name]
    return X
```

Call this once before solving to get `X0`.

### scatter_state_vector(X)

The inverse operation — writes values from X back to `comp._state_values`. Called inside the solver at every RHS evaluation so components see current state values:

```python
def scatter_state_vector(self, X: np.ndarray) -> None:
    for comp in self.components:
        off = self.state_offsets.get(comp.name)
        if off is not None:
            for i, name in enumerate(comp.state_names):
                comp._state_values[name] = X[off + i]
```

### all_state_names()

Returns a list of `"component.state"` strings in X-index order, used to populate `TransientSolution.state_names`:

```python
["chamber.P", "chamber.h", "rotor.omega", "pipe.mdot", ...]
```

---

## 3. Physics Models — Dynamic States

All dynamic states are integrated as ODEs: `dX/dt = f(t, X)`.

### 3.1 Volume — (P, h) states

**Source:** `atha/components/volume.py`

**States:** `P` [Pa], `h` [J/kg]

**Variables:**
- `V` — volume [m³]
- `m = ρ·V` — instantaneous mass [kg]
- `ρ`, `γ`, `T` — from `ThermoBackend.state_from_Ph(P, h)`
- `ṁ_in`, `ṁ_out` — inlet/outlet mass flows [kg/s]
- `h_in` — specific enthalpy of incoming flow [J/kg]
- `Q̇` — external heat input [W] (default 0)

**ODEs:**

```
dP/dt = (γ · R_eff · T / V) · (Σṁ_in − Σṁ_out) / ρ

dh/dt = (1/m) · (Q̇ + Σ(ṁ_in · h_in) − Σ(ṁ_out · h) − V · dP/dt)
```

where `R_eff = (γ−1)/γ · cp = cp − cv`.

**Why (P, h) not (ρ, u)?** Pressure is enormously sensitive to density for liquids (bulk modulus ~GPa). Using ρ as a state produces ill-conditioned Newton iterations. P and h are the natural potentials for flow networks and the primary inputs to `CoolProp.state_from_Ph()`.

**Implementation notes:**
- `compute_outputs` calls `thermo.state_from_Ph(P, h)` → `FluidState`
- `get_state_derivatives` assembles flux sums from the BCS dict using keys `"{port_name}.mdot"` and `"{port_name}.h"`
- Outlets use `h` (the volume's current enthalpy) as outflow enthalpy — well-mixed assumption
- `Q_dot` input key is optional (defaults to 0.0 if absent from BCS dict)

### 3.2 Rotor — ω state

**Source:** `atha/components/rotor.py`

**State:** `omega` [rad/s]

**Parameters:** `I` [kg·m²] moment of inertia, `k_f` [N·m·s/rad] viscous friction

**ODE:**

```
dω/dt = (τ_drive − τ_load − k_f · ω) / I
```

**Boundary condition keys:**
- `"shaft_in.tau"` — driving torque [N·m] (turbine, positive)
- `"shaft_out.tau"` — load torque [N·m] (pump, positive; internally negated)

**Steady state:** `ω_ss = (τ_drive − τ_load) / k_f`

### 3.3 PipeWithInertia — ṁ state

**Source:** `atha/components/pipe_inertia.py`

**State:** `mdot` [kg/s]

**Parameters:** `L` [m] length, `A` [m²] cross-section, `D` [m] diameter, `rho` [kg/m³] fluid density

**ODE:**

```
dṁ/dt = (A / L) · (P_in − P_out − ΔP_friction)
```

Friction loss (Darcy-Weisbach):
```
ΔP_friction = f · (L/D) · ρ · (ṁ / (ρ·A))² / 2
```

Friction factor `f` from Churchill's explicit approximation for all flow regimes.

**Boundary condition keys:**
- `"inlet.P"` — upstream pressure [Pa]
- `"outlet.P"` — downstream pressure [Pa]

### 3.4 MetalNode — T_wall state

**Source:** `atha/components/metal_node.py`

**State:** `T_wall` [K]

**Parameters:** `m_wall` [kg], `Cp_wall` [J/(kg·K)]

**ODE:**

```
dT_wall/dt = (Q̇_hot − Q̇_cool) / (m_wall · Cp_wall)
```

Heat inputs are provided via BCS dict keys `"hot_side.Q_dot"` and `"cool_side.Q_dot"`.

---

## 4. Physics Models — Algebraic Components

Algebraic components have no integration states. They compute outputs from inputs instantaneously.

### 4.1 PipeAlgebraic

Loss-coefficient pressure drop:

```
ΔP = K_loss · ρ · (ṁ / (ρ·A))² / 2
```

### 4.2 OrificeCompressible

**Choked condition:** `P_in / P_out ≥ ((γ+1)/2)^(γ/(γ−1))`

Subsonic mass flow:
```
ṁ = Cd · A · P_in · √(γ/(R·T_in)) ·
    √((2/(γ−1)) · [(P_out/P_in)^(2/γ) − (P_out/P_in)^((γ+1)/γ)])
```

Choked mass flow:
```
ṁ = Cd · A · P_in · √(γ/(R·T_in)) · (2/(γ+1))^((γ+1)/(2(γ−1)))
```

Branches blend smoothly near critical pressure ratio to maintain a continuous Jacobian.

### 4.3 Valve

Incompressible variable-area:
```
ṁ = Cv · A_frac · A_max · √(ρ · |ΔP|) · sign(ΔP)
```

`A_frac` ∈ [0, 1] is set via BCS dict key `"cmd.area_frac"`.

### 4.4 Nozzle

Isentropic nozzle, throat-choked flow. Outputs: thrust [N], exit pressure [Pa], exit velocity [m/s], Cf.

Throat mass flow:
```
ṁ = A_t · P_c · √(γ/(R·T_c)) · (2/(γ+1))^((γ+1)/(2(γ−1)))
```

Exit Mach solved from area-Mach relation via `scipy.optimize.brentq`. Thrust:
```
F = ṁ · V_e + (P_e − P_a) · A_e
```

### 4.5 Pump and Turbine

Map-based algebraic components (see Phase 6 in the implementation plan). Both read from 2-D head/efficiency maps interpolated at runtime. Pump produces `ΔP = ψ(φ) · ρ · N² · D²`; turbine produces `W = ṁ · η_t · Δh_s`.

---

## 5. Thermodynamic Backends

All backends implement `ThermoBackend` (`atha/thermo/interface.py`) and return a frozen `FluidState` dataclass.

### FluidState fields

| Field | Units | Description |
|-------|-------|-------------|
| `P` | Pa | Pressure |
| `T` | K | Temperature |
| `h` | J/kg | Specific enthalpy |
| `rho` | kg/m³ | Density |
| `s` | J/(kg·K) | Specific entropy |
| `cp` | J/(kg·K) | Constant-pressure specific heat |
| `cv` | J/(kg·K) | Constant-volume specific heat |
| `gamma` | — | cp/cv |
| `mu` | Pa·s | Dynamic viscosity |
| `k` | W/(m·K) | Thermal conductivity |
| `MW` | kg/mol | Mean molecular weight |
| `phase` | str | 'gas', 'liquid', 'two-phase', 'supercritical' |
| `quality` | float or None | Vapor quality (two-phase only) |

`FluidState` is frozen (`@dataclass(frozen=True)`) — any attempt to assign a field raises `FrozenInstanceError`.

### IdealGasBackend

`atha/thermo/ideal_gas.py` — analytic ideal gas for testing without external dependencies.

```python
gas = IdealGasBackend(gamma=1.4, R=287.0)   # air
gas = IdealGasBackend(gamma=1.24, R=711.0)  # LOX/LH2 products at MR=6
```

`state_from_PT(P, T)`:
- `rho = P / (R·T)`
- `h = cp·T`  (reference h=0 at T=0)
- `s = cp·ln(T) − R·ln(P)`

`isentropic_expansion(inlet, P_exit)`:
- `T_exit = T_in · (P_exit/P_in)^((γ−1)/γ)`

### CoolPropBackend

`atha/thermo/coolprop_backend.py` — real-fluid EOS via CoolProp.

```python
lox = CoolPropBackend("Oxygen")
lh2 = CoolPropBackend("Hydrogen")
lch4 = CoolPropBackend("Methane")
```

Primary hot path: `state_from_Ph(P, h)` → calls `AS.update(CP.HmassP_INPUTS, h, P)`.

**Phase detection:** `AS.phase()` returns `CP.iphase_liquid`, `iphase_gas`, `iphase_twophase`, etc. The backend maps these to human-readable strings and sets `quality = AS.Q()` for two-phase states.

**Performance tip:** For LOX/LH2 pump simulations, enable the BICUBIC tabular backend:
```python
lox = CoolPropBackend("Oxygen", backend="BICUBIC&HEOS")
```
This gives ~100× speedup after a one-time setup, critical for transient integration.

### CanteraBackend

`atha/thermo/cantera_backend.py` — combustion equilibrium and gas-phase properties.

```python
cb = CanteraBackend("gri30.yaml")
result = cb.combustion_equilibrium(fuel="H2", oxidizer="O2",
                                    MR=6.0, P_chamber=20e6)
# result.T_ad, result.c_star, result.gamma, result.MW
```

`combustion_equilibrium()` sets the mixture, calls `gas.equilibrate('HP')` (adiabatic constant-pressure), and computes c* from the product mixture properties.

**Caching note:** `equilibrate('HP')` takes 0.5–2 ms per call. For transient simulations with a combustion chamber, pre-compute a (MR, P) grid and fit a 2-D spline before integration. The `CombustionChamber` component handles this automatically.

---

## 6. Port System and Component Wiring

### Port domains

| Domain | Class | Quantities |
|--------|-------|-----------|
| FLUID | `FluidPort` | `mdot` [kg/s], `P` [Pa], `h` [J/kg] |
| SHAFT | `ShaftPort` | `omega` [rad/s], `tau` [N·m] |
| THERMAL | `ThermalPort` | `T_wall` [K], `Q_dot` [W] |

### Connection rules

`port.connect(other)` enforces:
1. **Same domain** — raises `ValueError("domain")` on mismatch (e.g., FluidPort → ShaftPort)
2. **Opposite direction** — raises `ValueError("direction")` if both are INLET or both OUTLET

### How BCS keys correspond to ports

Ports declared via `add_inlet("name")` or `add_outlet("name")` create named FluidPort objects. Inside `get_state_derivatives`, the component reads flow values from the BCS dict using `"{port_name}.mdot"` and `"{port_name}.h"` keys.

Example: a Volume with `add_inlet("fuel_in")` expects `bcs["fuel_in.mdot"]` and `bcs["fuel_in.h"]`.

For non-Volume components (Rotor, Pipe, etc.), the BCS key names are documented in each component's `__init__` docstring.

---

## 7. Steady-State Solver

**Source:** `atha/solver/steady_state.py`

### Algorithm

The steady-state problem is: find `X*` such that `dX/dt = 0` for all states and all algebraic residuals are zero.

This reduces to: `F(X*) = 0` where `F` concatenates all `get_state_derivatives()` and `get_residuals()` outputs across all components.

```python
def residuals(X):
    layout.scatter_state_vector(X)
    resid = []
    for comp in layout.components:
        off = layout.state_offsets.get(comp.name)
        states = {n: float(X[off+i]) for i,n in enumerate(comp.state_names)} if off else {}
        inputs = dict(boundary_conditions)   # constant BCS for steady state
        outputs = comp.compute_outputs(0.0, states, inputs)
        derivs  = comp.get_state_derivatives(0.0, states, inputs, outputs)
        for name in comp.state_names:
            resid.append(derivs.get(name, 0.0))
        resid.extend(comp.get_residuals(0.0, states, inputs, outputs).values())
    return np.array(resid) if resid else np.zeros(len(X))
```

### Underlying solver

`newton_solve()` calls `scipy.optimize.root(F, x0, method='hybr')`. The `'hybr'` method is Powell's hybrid method (a trust-region Newton variant), which is more robust than pure Newton-Raphson for ill-conditioned systems. Convergence tolerance is 1e-10 by default.

### Usage

```python
from atha.solver.steady_state import SteadyStateSolver

solver = SteadyStateSolver(layout, tol=1e-8, max_iter=200)
X_ss = solver.solve(X0, boundary_conditions_dict)
```

`boundary_conditions_dict` is a plain `Dict[str, float]` (not a callable). For steady-state, time does not vary.

After a successful solve, `layout.scatter_state_vector(X_ss)` has been called, so component `_state_values` reflect the solution.

### Initial guess quality

Powell's method requires a reasonable initial guess. For engine cycles:
- Set `initial_P` and `initial_T` in each Volume/CombustionChamber to expected operating values
- For rotors, start at expected steady-state speed rather than zero

A poor initial guess causes `RuntimeError("Newton solver failed: ...")`.

---

## 8. Transient Solver

**Source:** `atha/solver/transient.py`

### Algorithm

`TransientSolver.integrate()` wraps `scipy.integrate.solve_ivp` with the Radau implicit Runge-Kutta method (order 5, A-stable).

The RHS function:
```python
def rhs(t, X):
    layout.scatter_state_vector(X)
    bcs = boundary_conditions_fn(t)      # callable, evaluated at current t
    dXdt = np.zeros_like(X)
    for comp in layout.components:
        off = layout.state_offsets.get(comp.name)
        states = {n: float(X[off+i]) ...} if off else {}
        inputs = dict(bcs)
        outputs = comp.compute_outputs(t, states, inputs)
        derivs  = comp.get_state_derivatives(t, states, inputs, outputs)
        if off is not None:
            for i, n in enumerate(comp.state_names):
                dXdt[off + i] = derivs.get(n, 0.0)
    return dXdt
```

### Why Radau?

Rocket engine transients are inherently stiff. Acoustic wave propagation has a time constant of ~1 ms (speed of sound / pipe length). Thermal dynamics have time constants of ~10–100 s. The stiffness ratio is ~10⁵.

Explicit methods (RK45) are conditionally stable: they require step sizes ≤ 1 ms even when the interesting physics only changes on 10–100 ms scales, wasting 10–100× of computation.

Radau is A-stable (unconditionally stable for any step size) and adapts step size to accuracy, not stability. For this problem, Radau is typically 10–50× faster than RK45.

### Solver configuration

```python
solver = TransientSolver(
    layout,
    method="Radau",    # or "BDF" for very stiff systems
    rtol=1e-4,         # relative tolerance on state change per step
    atol=1e-6,         # absolute tolerance (SI units)
    max_step=0.05,     # cap step size [s]; smaller values capture fast valve events
)
```

**`max_step`** is critical when the boundary conditions change discontinuously (valve step, throttle ramp). Without a cap, Radau may take large steps that skip over the discontinuity. Set `max_step` to roughly 1/10 of the fastest BCS change timescale.

**`rtol` and `atol`**: these control step size selection. For pressures in the 10⁵–10⁷ Pa range, `atol=1e-6` is effectively zero tolerance — which is fine, since `rtol=1e-4` will dominate for large-magnitude states. If you see solver failures or unexpectedly slow integration, tighten `atol` to match the magnitude of your smallest state (e.g., `atol=1.0` for temperatures in hundreds of K).

### Boundary conditions callable

```python
def bcs(t: float) -> Dict[str, float]:
    return {"inlet.mdot": 0.1, "inlet.h": h_ref}
```

The callable must be fast (it's called O(1000) times per second of simulation) and must return a dict with all required keys for all active components. Missing keys default to 0.0 inside components that use `.get()`.

For time-varying commands:
```python
def bcs(t):
    # Ramp valve from 0 to 1 over 1 second
    area_frac = min(1.0, t / 1.0)
    return {"valve.cmd.area_frac": area_frac, "inlet.P": 5e6}
```

### TransientSolution

```python
@dataclass
class TransientSolution:
    t:           np.ndarray      # shape (N,), time points [s]
    X:           np.ndarray      # shape (N, n_states), state history
    state_names: List[str]       # ["comp.state", ...] matching X columns
```

Access a specific state time series:
```python
P = result.get("chamber", "P")      # np.ndarray shape (N,)
omega = result.get("rotor", "omega")
```

Access the full state matrix:
```python
result.X[:, 0]   # first state (check result.state_names[0] for identity)
```

SciPy's Radau uses adaptive step sizes, so `result.t` is non-uniform. For uniform-time output, interpolate:
```python
t_uniform = np.linspace(0, t_end, 1000)
P_uniform = np.interp(t_uniform, result.t, result.get("chamber", "P"))
```

Or use the dense output directly (enabled by `dense_output=True` in `solve_ivp`):
```python
# sol is the raw scipy OdeSolution; use result.t and result.X for ATHA interface
```

---

## 9. JANNAF Performance Module

**Source:** `atha/jannaf/`

### Efficiency factors

```python
from atha.jannaf.efficiency import JANNAFEfficiencies

eff = JANNAFEfficiencies(
    eta_cstar=0.975,         # combustion efficiency (c*_del / c*_ideal)
    eta_Cd=0.98,             # discharge coefficient (actual ṁ / ideal ṁ)
    eta_velocity=0.99,       # viscous nozzle losses
    eta_divergence=0.9830,   # divergence loss: 0.5·(1 + cos α), α=15° conical
    eta_two_phase=1.0,       # two-phase flow losses (condensation)
    eta_boundary_layer=0.99, # boundary layer momentum deficit
)
```

Default `eta_divergence=0.9830` corresponds to a 15° half-angle conical nozzle:
`η_div = ½·(1 + cos 15°) = 0.9830`.

### SimplifiedJANNAF

```python
from atha.jannaf.simplified import SimplifiedJANNAF

jannaf = SimplifiedJANNAF(
    thermo=thermo_backend,
    efficiencies=eff,
    throat_area=A_t,        # m²
    exit_area=A_t * eps,    # m²; eps = expansion ratio Ae/At
    ambient_pressure=0.0,   # Pa; 0.0 for vacuum Isp
)

result = jannaf.compute(
    P_chamber=Pc,           # Pa
    T_chamber=T_ad,         # K (adiabatic flame temperature)
    MR=6.0,                 # O/F mass ratio
    mdot_total=468.0,       # kg/s total propellant flow
)
```

### Calculation chain

```
1. state = thermo.state_from_PT(Pc, Tc)  → γ, R_eff

2. c*_ideal = √(γ·R·T) / [γ · √((2/(γ+1))^((γ+1)/(γ−1)))]

3. c*_del = η_c* · c*_ideal

4. Solve area-Mach relation for Me (scipy.optimize.brentq on supersonic branch)
   f(M) = (1/M)·[(2/(γ+1))·(1 + (γ−1)/2·M²)]^((γ+1)/(2(γ−1))) − ε

5. Pe = Pc · (1 + (γ−1)/2·Me²)^(−γ/(γ−1))

6. Cf_ideal = √(2γ²/(γ−1) · (2/(γ+1))^((γ+1)/(γ−1)) · [1−(Pe/Pc)^((γ−1)/γ)])
              + (Pe − Pa)/Pc · ε

7. Cf_del = η_velocity · η_divergence · η_boundary_layer · Cf_ideal

8. Isp_del = c*_del · Cf_del / g0   (g0 = 9.80665 m/s²)

9. thrust  = Cf_del · Pc · At
```

### PerformanceResult

```python
result.Isp           # s, delivered specific impulse
result.c_star        # m/s, delivered characteristic velocity
result.Cf            # dimensionless, delivered thrust coefficient
result.thrust        # N, delivered thrust
result.c_star_ideal  # m/s, ideal c*
result.Isp_ideal     # s, ideal Isp
result.P_exit        # Pa, nozzle exit pressure
result.V_exit        # m/s, exit velocity
result.expansion_ratio  # Ae/At
result.efficiencies  # JANNAFEfficiencies dataclass
```

---

## 10. Setting Up a Simulation — Step-by-Step

### Scenario A: Single volume, transient pressure rise

```python
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
from atha.solver.transient import TransientSolver

# 1. Create thermod backend
gas = IdealGasBackend(gamma=1.4, R=287.0)

# 2. Create components with initial conditions
vol = Volume("tank", volume=0.1,     # m³
             thermo=gas,
             initial_P=1e5,          # Pa, 1 bar
             initial_T=300.0)        # K

# 3. Declare ports (create inlet/outlet port objects)
vol.add_inlet("fill")

# 4. Assemble engine
engine = Engine("pressure_rise")
engine.add_component(vol)

# 5. Compile → EngineLayout with index assignments
layout = engine.compile()

# 6. Build initial state vector from component initial values
X0 = layout.assemble_state_vector()

# 7. Define boundary conditions (time-varying callable)
fs_ref = gas.state_from_PT(1e5, 300.0)
def bcs(t):
    return {"fill.mdot": 0.05,    # kg/s constant inflow
            "fill.h": fs_ref.h}   # J/kg enthalpy of incoming gas

# 8. Create solver and integrate
solver = TransientSolver(layout, method="Radau", max_step=0.1)
result = solver.integrate(t_span=(0.0, 10.0), X0=X0, boundary_conditions_fn=bcs)

# 9. Extract results
P = result.get("tank", "P")
print(f"Pressure: {P[0]/1e5:.3f} → {P[-1]/1e5:.3f} bar over {result.t[-1]:.1f} s")
```

### Scenario B: Rotor spin-up with steady-state check

```python
from atha.components.rotor import Rotor
from atha.solver.steady_state import SteadyStateSolver

rotor = Rotor("turbopump", inertia=10.0, friction_coeff=0.5, initial_omega=0.0)

engine = Engine("spinup")
engine.add_component(rotor)
layout = engine.compile()
X0 = layout.assemble_state_vector()

# Transient spin-up
def bcs(t):
    return {"shaft_in.tau": 500.0,   # N·m drive torque from turbine
            "shaft_out.tau": 100.0}  # N·m load from pump

solver = TransientSolver(layout, method="Radau", max_step=0.05)
result = solver.integrate((0.0, 60.0), X0, bcs)
omega = result.get("turbopump", "omega")

# Theoretical steady state: omega_ss = (500 - 100) / 0.5 = 800 rad/s
print(f"Final omega: {omega[-1]:.1f} rad/s  (theoretical: 800.0)")

# Verify with steady-state solver
bcs_ss = {"shaft_in.tau": 500.0, "shaft_out.tau": 100.0}
X_ss = SteadyStateSolver(layout).solve(X0 * 0 + 800.0, bcs_ss)
layout.scatter_state_vector(X_ss)
print(f"Steady-state omega: {layout.components[0]._state_values['omega']:.6f} rad/s")
```

### Scenario C: Two coupled volumes

```python
# Two tanks with a flow connection between them
gas = IdealGasBackend(gamma=1.4, R=287.0)

v1 = Volume("high_P", volume=0.05, thermo=gas, initial_P=5e5, initial_T=300.0)
v2 = Volume("low_P",  volume=0.05, thermo=gas, initial_P=1e5,  initial_T=300.0)

# v1 has an outlet; v2 has an inlet (the orifice between them is modeled via BCS)
v1.add_outlet("transfer_out")
v2.add_inlet("transfer_in")

engine = Engine("two_tank")
engine.add_component(v1)
engine.add_component(v2)
layout = engine.compile()
X0 = layout.assemble_state_vector()

def bcs(t):
    # In a real simulation the orifice component would compute mdot from ΔP.
    # For this example, impose a fixed transfer flow:
    return {
        "transfer_out.mdot": 0.01,   # kg/s leaving v1
        "transfer_in.mdot":  0.01,   # kg/s entering v2 (same value — conservation)
        "transfer_in.h": 287.0 * 1.4/0.4 * 300.0,  # enthalpy of transferred gas
    }

solver = TransientSolver(layout, method="Radau", max_step=0.1)
result = solver.integrate((0.0, 20.0), X0, bcs)

P1 = result.get("high_P", "P")
P2 = result.get("low_P",  "P")
```

### Scenario D: JANNAF nozzle performance sweep

```python
import numpy as np
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.thermo.ideal_gas import IdealGasBackend

thermo = IdealGasBackend(gamma=1.24, R=711.0)   # LOX/LH2 approx
eff = JANNAFEfficiencies(eta_cstar=0.975, eta_Cd=0.98,
                          eta_velocity=0.99, eta_divergence=0.9830)

A_t = 0.0687   # m², throat area

# Sweep expansion ratio from 10 to 100
eps_values = np.linspace(10, 100, 20)
Isp_values = []

for eps in eps_values:
    jannaf = SimplifiedJANNAF(thermo=thermo, efficiencies=eff,
                               throat_area=A_t, exit_area=A_t * eps,
                               ambient_pressure=0.0)
    r = jannaf.compute(P_chamber=20.6e6, T_chamber=3560.0,
                        MR=6.0, mdot_total=468.0)
    Isp_values.append(r.Isp)

# Isp_values increases with expansion ratio (vacuum nozzle)
```

---

## 11. Extracting and Interpreting Results

### TransientSolution

```python
result.t                          # np.ndarray, shape (N,), time points [s]
result.X                          # np.ndarray, shape (N, n_states), full state history
result.state_names                # List[str], e.g. ["tank.P", "tank.h", "rotor.omega"]

# Named access
P = result.get("tank", "P")       # np.ndarray, shape (N,)
h = result.get("tank", "h")       # np.ndarray, shape (N,)
omega = result.get("rotor", "omega")

# Time of a specific event (e.g., pressure reaches 3 bar)
idx = np.argmax(P > 3e5)
t_event = result.t[idx]

# Rate of change at end of simulation
dP_dt_final = np.gradient(P, result.t)[-1]

# Check for steady state (all dX/dt < threshold)
is_ss = np.all(np.abs(np.gradient(result.X, result.t, axis=0)[-1]) < 1.0)
```

### Steady-state output

`SteadyStateSolver.solve()` returns `X_ss` (numpy array) and also calls `scatter_state_vector(X_ss)` so component `_state_values` are updated:

```python
X_ss = solver.solve(X0, bcs)

# Read from layout
layout.scatter_state_vector(X_ss)   # already called by solver, but safe to repeat
for comp in layout.components:
    off = layout.state_offsets.get(comp.name)
    if off is not None:
        for i, name in enumerate(comp.state_names):
            print(f"  {comp.name}.{name} = {X_ss[off+i]:.4g}")
```

### PerformanceResult (JANNAF)

```python
r = jannaf.compute(...)

print(f"Isp (del):        {r.Isp:.2f} s")
print(f"Isp (ideal):      {r.Isp_ideal:.2f} s")
print(f"c* (del):         {r.c_star:.2f} m/s")
print(f"c* (ideal):       {r.c_star_ideal:.2f} m/s")
print(f"Thrust:           {r.thrust/1000:.2f} kN")
print(f"Cf (del):         {r.Cf:.4f}")
print(f"Expansion ratio:  {r.expansion_ratio:.1f}")
print(f"Exit pressure:    {r.P_exit/1e3:.2f} kPa")
print(f"Exit velocity:    {r.V_exit:.1f} m/s")
print(f"  η_c*={r.efficiencies.eta_cstar}, η_Cd={r.efficiencies.eta_Cd}")
```

### TTBEResult

```python
from atha.validation.ttbe import run_ttbe_100pct_rpl

r = run_ttbe_100pct_rpl()
print(f"Isp_vacuum:    {r.Isp_vacuum:.2f} s")
print(f"Thrust_vacuum: {r.thrust_vacuum/1000:.2f} kN")
print(f"c_star:        {r.c_star:.2f} m/s")
print(f"Cf_vacuum:     {r.Cf_vacuum:.4f}")
print(r.performance)   # full PerformanceResult
```

---

## 12. Component Reference

| Class | Module | States | Key BCS keys |
|-------|--------|--------|--------------|
| `Volume` | `components/volume.py` | P [Pa], h [J/kg] | `{port}.mdot`, `{port}.h`, `Q_dot` |
| `Rotor` | `components/rotor.py` | omega [rad/s] | `shaft_in.tau`, `shaft_out.tau` |
| `PipeWithInertia` | `components/pipe_inertia.py` | mdot [kg/s] | `inlet.P`, `outlet.P` |
| `MetalNode` | `components/metal_node.py` | T_wall [K] | `hot_side.Q_dot`, `cool_side.Q_dot` |
| `PipeAlgebraic` | `components/pipe_algebraic.py` | — (algebraic) | `inlet.P`, `inlet.mdot` |
| `OrificeCompressible` | `components/orifice.py` | — (algebraic) | `inlet.P`, `inlet.T`, `outlet.P` |
| `Valve` | `components/valve.py` | — (algebraic) | `inlet.P`, `outlet.P`, `cmd.area_frac` |
| `Nozzle` | `components/nozzle.py` | — (algebraic) | `inlet.P`, `inlet.T`, `ambient.P` |
| `Pump` | `components/pump.py` | — (algebraic) | `inlet.P`, `inlet.h`, `shaft.omega` |
| `Turbine` | `components/turbine.py` | — (algebraic) | `inlet.P`, `inlet.h`, `outlet.P` |

### Constructor arguments

**Volume:**
```python
Volume(name: str, volume: float,         # m³
       thermo: ThermoBackend,
       initial_P: float = 1e5,           # Pa
       initial_T: float = 300.0)         # K
```

**Rotor:**
```python
Rotor(name: str, inertia: float,         # kg·m²
      friction_coeff: float,             # N·m·s/rad
      initial_omega: float = 0.0)        # rad/s
```

**PipeWithInertia:**
```python
PipeWithInertia(name: str, length: float,     # m
                area: float,                  # m²
                diameter: float,              # m
                fluid_density: float,         # kg/m³
                initial_mdot: float = 0.0)    # kg/s
```

---

## 13. Boundary Conditions Dictionary Patterns

The BCS dict is a flat `Dict[str, float]` passed to every component at every evaluation.

### Key naming convention

```
"{port_name}.{quantity}"
```

Examples:
- `"inlet.mdot"` — mass flow into an inlet named "inlet" [kg/s]
- `"inlet.h"` — specific enthalpy of incoming flow [J/kg]
- `"outlet.mdot"` — mass flow from an outlet named "outlet" [kg/s]
- `"shaft_in.tau"` — driving torque [N·m]
- `"shaft_out.tau"` — load torque [N·m]

### Multi-port volumes

```python
vol.add_inlet("oxidizer_in")
vol.add_inlet("fuel_in")
vol.add_outlet("nozzle_out")

def bcs(t):
    return {
        "oxidizer_in.mdot": mdot_ox,
        "oxidizer_in.h": h_ox,
        "fuel_in.mdot": mdot_fuel,
        "fuel_in.h": h_fuel,
        "nozzle_out.mdot": mdot_throat,   # computed by nozzle model
    }
```

### Missing keys

Components use `inputs.get("key", 0.0)` for optional inputs. Missing BCS keys return 0.0 — useful when a port exists but no flow is connected yet. If a required key is missing and the component does not have a default, it raises `KeyError` with a descriptive message.

### Time-varying commands

```python
def bcs(t):
    # Throttle ramp: 100% to 65% RPL over 5 seconds starting at t=2s
    if t < 2.0:
        RPL = 1.0
    elif t < 7.0:
        RPL = 1.0 - 0.35 * (t - 2.0) / 5.0
    else:
        RPL = 0.65

    mdot_total = 468.0 * RPL
    return {
        "ox_in.mdot": mdot_total * 6.0/7.0,
        "ox_in.h": h_ox_ref,
        "fuel_in.mdot": mdot_total * 1.0/7.0,
        "fuel_in.h": h_fuel_ref,
    }
```

---

## 14. Numerical Considerations and Failure Modes

### Integration failure: "Transient integration failed"

Raised when `solve_ivp` returns `sol.success == False`. Common causes:

1. **Singular ODE** — a state goes to zero or negative (e.g., a Volume loses all mass). Reduce `max_step` or constrain flow rates.

2. **Stiff BCS discontinuity** — a step change in BCS at time `t` causes very large `dX/dt`. Reduce `max_step` around the event, or ramp the BCS instead.

3. **Inconsistent initial conditions** — `X0` places a component far from a physically valid state (e.g., `h < 0` for an ideal gas). Verify initial `P` and `T` produce a valid `FluidState`.

4. **Too-tight tolerances** — `rtol=1e-8` or `atol=1e-12` for this problem will cause the solver to fail trying to achieve machine precision. Keep `rtol=1e-4`, `atol=1e-6`.

### Newton solver failure: "Newton solver failed"

1. **Poor initial guess** — set component initial conditions to expected operating point values.

2. **Circular algebraic dependencies** — staged combustion engines with tightly coupled preburner and main chamber may have residuals that depend on each other. Add an explicit `inner_newton` solve for algebraic variables.

3. **Discontinuous residuals** — a component with `if/else` on a state value creates a discontinuous Jacobian. Use smooth blends (e.g., hyperbolic tangent).

### JANNAF area-Mach root-find failure

`_exit_mach` uses `brentq` on the supersonic branch `[1.001, 100.0]`. This fails if:
- `epsilon <= 1.0` — an expansion ratio of exactly 1 means the nozzle has no expansion. Guard: `if epsilon <= 1.0 + 1e-6: return 1.0`
- `epsilon > 1000` — Mach numbers above ~100 are unphysical; upper bound may need expansion

### CoolProp near phase boundaries

CoolProp throws `ValueError` or returns NaN near the critical point. Check `FluidState.phase` before using properties. For LOX pumps, verify the fluid is in the `'liquid'` phase (subcooled) before each call.

### Ideal gas reference enthalpy

`IdealGasBackend` uses `h = cp·T` (zero reference at T=0 K). This means enthalpy values are always positive. When mixing ideal gas components with CoolProp components, be aware that CoolProp uses a different enthalpy reference (typically ASHRAE or NBP). Do not compare absolute enthalpy values across backends — only differences (Δh) are physically meaningful across reference conventions.

### Volume mass depletion

A Volume with only an outlet and no inlet will drain to zero mass. Near `m → 0`, `dh/dt` diverges (division by `m`). Either:
- Set a minimum mass floor in the component
- Add an event function to `TransientSolver.integrate(events=...)` that stops integration when `P < P_min`
- Ensure inflow ≥ outflow for the simulation duration
