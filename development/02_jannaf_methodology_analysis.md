# JANNAF Rocket Engine Performance Prediction Manual Analysis
## Source: JANNAF ROCKET ENGINE.pdf — CPIA Publication 246, April 1975
## Chemical Propulsion Information Agency, Johns Hopkins APL

---

## Purpose

Standardized methodology for analytical prediction and evaluation of liquid rocket
engine performance. Developed by the JANNAF (Joint Army-Navy-NASA-Air Force)
Performance Standardization Working Group. Extends from thrust chamber to full engine.

---

## Two Main Procedures

### 1. Rigorous Analytical Procedure (Section 2)

Step-by-step procedure for highest accuracy:

1. **System Specification** — define geometry, propellants, operating conditions
2. **Injector Flow Distribution** — per-element mass flow, injection velocities, pressure drops
3. **Injector Analysis** — mixing quality, local equivalence ratios per stream tube
4. **Turbulent Combustion** — mixing efficiency, characteristic mixing length
5. **Stream Tube Combustion Analysis** — axial combustion profile, local T and composition
6. **Boundary Layer Analysis** — displacement thickness, wall heat flux, momentum deficit
7. **Wall Response and Iteration** — wall temperature convergence
8. **Inviscid Flow Analysis** — Method of Characteristics (MOC) or 1D isentropic nozzle
9. **Performance Calculation** — integrate to get thrust, Isp, c*
10. **Uncertainty Analysis** — propagate measurement and model uncertainties

### 2. Simplified Analytical Procedure (Section 3)

Uses efficiency factors applied to ideal (CEA-equivalent) performance:

```
c*_delivered = η_c* × c*_ideal

ṁ_actual = η_Cd × ṁ_ideal

Cf_delivered = η_velocity × η_divergence × η_boundary_layer × Cf_ideal

Isp_delivered = c*_delivered × Cf_delivered / g₀
```

**Efficiency factor definitions:**

| Symbol | Name | Typical Range | Definition |
|--------|------|--------------|------------|
| η_c* | Combustion efficiency | 0.94–0.99 | Actual c* / Ideal c* |
| η_Cd | Discharge coefficient | 0.96–0.99 | Actual ṁ / Ideal ṁ through throat |
| η_v | Velocity coefficient | 0.97–0.995 | Accounts for viscous losses |
| η_div | Divergence efficiency | 0.97–0.99 | ½(1 + cos α) for conical nozzle |

For a 15° half-angle conical nozzle: η_div = ½(1 + cos 15°) = 0.9830

---

## Ideal Performance Reference Formulas

### Characteristic Velocity (c*)
```
c*_ideal = √(R*T_c / γ) × √((γ+1)/2)^((γ+1)/(γ-1)) / √(2γ/(γ-1))

Simplified: c*_ideal = P_c * A_t / ṁ    (from continuity at throat)
```

### Thrust Coefficient (Cf)
```
Cf_ideal = √(2γ²/(γ-1) × (2/(γ+1))^((γ+1)/(γ-1)) × [1-(Pe/Pc)^((γ-1)/γ)])
           + (Pe - Pa)/Pc × Ae/At
```

### Specific Impulse
```
Isp = c* × Cf / g₀    where g₀ = 9.80665 m/s²
```

### Thrust
```
F = Cf × Pc × At = ṁ × Isp × g₀
```

---

## Ideal Nozzle Expansion (Isentropic 1D)

Area-Mach relationship:
```
Ae/At = (1/M) × [(2/(γ+1)) × (1 + (γ-1)/2 × M²)]^((γ+1)/(2(γ-1)))
```

Pressure ratio:
```
Pe/Pc = [1 + (γ-1)/2 × Me²]^(-γ/(γ-1))
```

Exit velocity:
```
Ve = Me × √(γ × R × Te)
```

---

## Standard Computer Programs Referenced (Section 5)

- **ODE/TDK** (One-Dimensional Equilibrium / Two-Dimensional Kinetics) — primary performance prediction code
- **BLIMPJ** — Boundary Layer Integral Matrix Procedure (modified for rockets)
- **VIPER** — Viscous Interaction Performance Evaluation Routine

Note: All 1970s FORTRAN codes. Cantera + custom Python replaces these in ATHA.

---

## Thermochemical Properties (Section 6)

JANNAF uses NASA 7-coefficient polynomial for Cp(T):
```
Cp/R = a1 + a2*T + a3*T² + a4*T³ + a5*T⁴
h/(RT) = a1 + a2*T/2 + a3*T²/3 + a4*T³/4 + a5*T⁴/5 + a6/T
s/R = a1*ln(T) + a2*T + a3*T²/2 + a4*T³/3 + a5*T⁴/4 + a7
```
Cantera uses this exact format — direct compatibility.

---

## Uncertainty Analysis

- Each efficiency factor has a 1σ uncertainty band (typically 0.3–1.0%)
- Combined Isp uncertainty: RSS of individual contributions
- Typical delivered Isp uncertainty: ±0.5–1.5% (1σ) for well-characterized designs
