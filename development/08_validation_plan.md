# Validation Plan

---

## Tier 1: Unit-Level Validation

Each component validated against analytical solutions.

### Nozzle (IdealGasBackend, γ=1.2, R=520 J/(kg·K))
- Input: Pc=20MPa, Tc=3600K, At=0.0687m², Ae/At=77, vacuum
- Expected: Isp ~ 430-480s (approximate LOX/LH2 range)
- Check: Cf > 0, c* > 0, thrust > 0

### Isentropic Nozzle Relations
- Verify exit Mach, pressure, temperature for γ=1.2, 1.3, 1.4
- Compare to NACA 1135 table values at known expansion ratios

### Volume Dynamics
- Step change in inlet flow to a volume
- Expected: P rises exponentially with time constant τ = V/(γRT/P × dṁ)
- Verify numerical integration matches analytical solution within 0.1%

### Rotor Dynamics
- Net torque 100 N⋅m on 10 kg⋅m² rotor (no friction)
- Expected: dω/dt = 10 rad/s²
- Integrate 1 second: ω should reach 10 rad/s

---

## Tier 2: Component Pair Validation

### Pump-Turbine Power Balance
- Connect pump to turbine on same shaft
- Steady state: turbine power = pump power + friction losses
- Verify rotor accelerates/decelerates correctly

### Chamber-Nozzle Thrust
- Connect CombustionChamber to Nozzle
- Verify: ṁ through nozzle = ṁ injected, thrust consistent with F = ṁ × Isp × g₀

---

## Tier 3: Simple Engine Cycle Validation

### Pressure-Fed Engine
- LOX/LH2 pressure-fed engine
- Steady-state: compare Isp to RPA reference
- Transient: valve step from 0→100%, verify P rise time constant

### Gas Generator Cycle
- Based on published publicly-available engine parameters
- Validate overall power balance: turbine power ≥ pump power

---

## Tier 4: TTBE Validation (Primary Validation Case)

### Target Agreement: within 3% for all parameters

The ROCKETS document provides TTBE detailed model comparison data vs. DTM.
ATHA should match ROCKETS results (which themselves agree with DTM to ~3%).

### TTBE Steady-State Validation Parameters
- Main chamber pressure (Pc)
- Turbine speeds: LPFT, HPFT, LPOT, HPOT
- Pump outlet pressures
- Preburner temperatures (fuel-rich and ox-rich)
- Overall Isp and thrust at 100% RPL

### TTBE Transient Validation
- Throttle transient: 100% RPL → 65% RPL → 100% RPL
- Compare response shape and settling time to ROCKETS figures
- Monitor: Pc, shaft speeds, preburner temperatures

---

## Tier 5: Regression Test Suite

Run after every significant code change:
```bash
pytest tests/ -v --tb=short
```

Key regression assertions:
1. TTBE steady-state: all parameters within 1% of baseline
2. Throttle transient: time traces within 2% of baseline
3. Linearization: all eigenvalues of A matrix have Re(λ) < 0 (stable at trim)
4. JANNAF Isp: within 0.5% of CEA reference at standard conditions
