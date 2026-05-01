# Physics Models Reference

All equations implemented in ATHA. Units are SI throughout (Pa, K, J/kg, kg/s, rad/s, N, m).

---

## 1. Lumped Volume (P, h states)

States: P [Pa], h [J/kg]
Volume: V [m³], mass: m = ρV [kg]

```
dP/dt = (γ_eff × R_eff × T / V) × (Σṁ_in - Σṁ_out)

dh/dt = (1/m) × (Q̇_net + Σ(ṁ_in × h_in) - Σ(ṁ_out × h_out) - V × dP/dt)
```

γ_eff, R_eff obtained from ThermoBackend at current (P, h).

---

## 2. Flow Inertia (ṁ state)

State: ṁ [kg/s]
Pipe: length L [m], cross-section area A [m²], diameter D [m]

```
dṁ/dt = (A/L) × (P_in - P_out - ΔP_friction - ΔP_gravity)

ΔP_friction = f × (L/D) × ρ × (ṁ/(ρA))² / 2    (Darcy-Weisbach)
```

Friction factor f from Moody chart:
- Laminar (Re < 2300): f = 64/Re
- Turbulent (Re > 4000): Churchill explicit approximation for f(Re, ε/D)

---

## 3. Rotor Speed Dynamics

State: ω [rad/s]
Moment of inertia: I [kg⋅m²]

```
dω/dt = (Στ_drive - Στ_load - τ_friction) / I

τ_friction = k_f × ω    (viscous bearing friction)
```

τ_drive: turbine torque (positive, delivers power to shaft)
τ_load:  pump torque   (negative, consumes power from shaft)

---

## 4. Metal Wall Temperature

State: T_wall [K]
Thermal mass: m_wall [kg], Cp_wall [J/(kg⋅K)]

```
dT_wall/dt = (Q̇_hot - Q̇_cool) / (m_wall × Cp_wall)

Q̇_hot  = h_hot  × A_hot  × (T_gas   - T_wall)     [W]
Q̇_cool = h_cool × A_cool × (T_wall  - T_coolant)   [W]
```

Bartz correlation for hot-side h_hot:
```
h_hot = (0.026/D_t^0.2) × (μ^0.2 × Cp/Pr^0.6) × (P_c × g_c/c*)^0.8 × (D_t/R_c)^0.1 × (A_t/A)^0.9 × σ
```

---

## 5. Pump (Algebraic)

Performance map: flow coefficient φ = ṁ/(ρ × N × D³)
Output:          head coefficient ψ = ΔH/(N² × D²)
                 efficiency η_p = f(φ)

```
ΔP = ρ × g × H = ρ × ψ(φ) × N² × D²

W_pump = ṁ × ΔP / (ρ × η_p)    [W]

τ_pump = W_pump / ω              [N⋅m]  (load on shaft)
```

---

## 6. Turbine (Algebraic)

Speed ratio: u/c₀ = π × D × N / √(2 × Δh_s)
Corrected flow: ṁ_corr = ṁ × √T_in / P_in

From map: η_t(PR, u/c₀) where PR = P_in/P_out

```
Δh_actual = η_t × Δh_s_isentropic

W_turb = ṁ × Δh_actual    [W]

τ_turb = W_turb / ω        [N⋅m]  (drives shaft, positive)
```

Isentropic enthalpy drop:
```
Δh_s = Cp × T_in × [1 - (P_out/P_in)^((γ-1)/γ)]
```

---

## 7. Compressible Orifice / Injector (Algebraic)

Choked condition: P_in/P_out ≥ ((γ+1)/2)^(γ/(γ-1))

```
# Subsonic:
ṁ = Cd × A × P_in × √(γ/(R × T_in)) ×
    √((2/(γ-1)) × [(P_out/P_in)^(2/γ) - (P_out/P_in)^((γ+1)/γ)])

# Choked:
ṁ = Cd × A × P_in × √(γ/(R × T_in)) × (2/(γ+1))^((γ+1)/(2(γ-1)))
```

Both branches use smooth hyperbolic blend at the choked transition.

---

## 8. Incompressible Valve (Algebraic)

```
ṁ = Cv × A_frac × A_max × √(ρ × |P_in - P_out|) × sign(P_in - P_out)
```

For variable area: A_frac(t) ∈ [0, 1] via command schedule.

---

## 9. Isentropic Nozzle Flow (Algebraic)

Throat condition (sonic):
```
ṁ = A_t × P_c × √(γ/(R × T_c)) × (2/(γ+1))^((γ+1)/(2(γ-1)))
```

Exit Mach from area-Mach relation (solved numerically via Brent's method):
```
Ae/At = (1/Me) × [(2/(γ+1)) × (1 + (γ-1)/2 × Me²)]^((γ+1)/(2(γ-1)))
Pe = Pc × [1 + (γ-1)/2 × Me²]^(-γ/(γ-1))
Ve = Me × √(γ × R × Te)
```

Thrust:
```
F = ṁ × Ve + (Pe - Pa) × Ae
```

---

## 10. JANNAF Thrust Coefficient (Analytical)

```
Cf_ideal = √(2γ²/(γ-1) × (2/(γ+1))^((γ+1)/(γ-1)) × [1-(Pe/Pc)^((γ-1)/γ)])
           + (Pe - Pa)/Pc × ε

Cf_delivered = η_v × η_div × η_bl × Cf_ideal

Isp = c*_delivered × Cf_delivered / g₀    where g₀ = 9.80665 m/s²
```
