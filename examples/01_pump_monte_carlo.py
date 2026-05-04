"""
Single Pump Monte Carlo Study
==============================
Propagates uncertainty in pump inlet conditions, shaft speed, and map
efficiency scaling through a single LOX pump to quantify outlet pressure
and efficiency variability.

Uses:
  - ComponentRig  for single-component evaluation
  - MonteCarloRunner with LHS sampling
  - SweepResult for performance map comparison

NOTE: Requires planned components — Pump, ComponentRig, ComponentSweep.
      PumpMap, CoolPropBackend, and MonteCarloRunner are implemented/planned.
"""

import numpy as np
from atha.components.pump import Pump, PumpMap
from atha.thermo.coolprop_backend import CoolPropBackend
from atha.analysis import ComponentRig, ComponentSweep, SweepAxis
from atha.monte_carlo import MonteCarloRunner, UncertainParameter, ParameterType

# ---------------------------------------------------------------------------
# Propellant
# ---------------------------------------------------------------------------
lox = CoolPropBackend("Oxygen")

# LOX inlet: subcooled at tank pressure, 91 K
P_inlet   = 3e5          # Pa   — tank pressure + NPSH margin
T_inlet   = 91.0         # K    — slightly subcooled relative to NBP 90.2 K
h_inlet   = lox.state_from_PT(P_inlet, T_inlet).h

# ---------------------------------------------------------------------------
# Pump definition (small LOX turbopump, ~5 kN engine)
#   Design point: 1.2 kg/s, 18 000 rpm, head rise 27 bar, η = 0.68
# ---------------------------------------------------------------------------
lox_pump_map = PumpMap.from_design_point(
    mdot_design=1.2,          # kg/s
    dP_design=2.7e6,          # Pa   — 27 bar head rise
    speed_design=18000,       # rpm
    efficiency_design=0.68,
    fluid_density=1141.0,     # kg/m³  LOX at inlet
)

pump = Pump(
    name="lox_pump",
    diameter=0.08,            # m  — impeller diameter
    pump_map=lox_pump_map,
    fluid=lox,
)

rig = ComponentRig(pump)

print("Required BCS keys:", rig.required_inputs())
# ['inlet.P', 'inlet.h', 'inlet.mdot', 'omega']

# ---------------------------------------------------------------------------
# 1. Single operating-point evaluation
# ---------------------------------------------------------------------------
nominal_bcs = {
    "inlet.P":    P_inlet,
    "inlet.h":    h_inlet,
    "inlet.mdot": 1.2,
    "omega":      18000 * np.pi / 30,   # rpm → rad/s
}

op = rig.evaluate(nominal_bcs)
print(f"\nNominal operating point:")
print(f"  Outlet pressure : {op['outlet.P']/1e5:.2f} bar")
print(f"  Head rise       : {(op['outlet.P'] - P_inlet)/1e5:.2f} bar")
print(f"  Efficiency      : {op['eta']:.3f}")
print(f"  Shaft power     : {op['power']/1e3:.2f} kW")
print(f"  Torque          : {op['tau']:.2f} N·m")

# ---------------------------------------------------------------------------
# 2. Monte Carlo — uncertain inlet conditions and shaft speed
# ---------------------------------------------------------------------------
params = [
    # LOX inlet temperature: ±2 K around nominal (tank conditioning variability)
    UncertainParameter("T_inlet",   ParameterType.NORMAL,  mean=91.0,   std=0.5),
    # LOX inlet pressure: ±0.1 bar (NPSH uncertainty)
    UncertainParameter("P_inlet",   ParameterType.NORMAL,  mean=3e5,    std=0.5e4),
    # Shaft speed: ±300 rpm (speed controller tolerance)
    UncertainParameter("speed_rpm", ParameterType.NORMAL,  mean=18000,  std=300),
    # Mass flow: ±3% (injector orifice Cd uncertainty)
    UncertainParameter("mdot",      ParameterType.NORMAL,  mean=1.2,    std=0.036),
    # Pump efficiency scaling: ±2% (map-to-hardware uncertainty)
    UncertainParameter("eta_scale", ParameterType.NORMAL,  mean=1.0,    std=0.02),
]

def evaluate_pump(X: dict) -> float:
    """Return outlet pressure [Pa] for one Monte Carlo sample."""
    h = lox.state_from_PT(X["P_inlet"], X["T_inlet"]).h
    pump.map_efficiency_scale = X["eta_scale"]   # mutable scaling factor
    result = rig.evaluate({
        "inlet.P":    X["P_inlet"],
        "inlet.h":    h,
        "inlet.mdot": X["mdot"],
        "omega":      X["speed_rpm"] * np.pi / 30,
    })
    return result["outlet.P"]

names = [p.name for p in params]

def evaluate_pump_row(X: np.ndarray) -> float:
    sample = {names[i]: float(X[i]) for i in range(len(names))}
    return evaluate_pump(sample)

runner = MonteCarloRunner(
    params=params,
    evaluate_fn=evaluate_pump_row,
    n_samples=1000,
    sampler="lhs",
    n_jobs=-1,
    seed=42,
)
mc_result = runner.run()
mc_result.print_summary()
# MonteCarloResult — 1000 samples
#   mean     : 27.3 bar
#   std      :  0.8 bar
#   CV       :  3.0 %
#   5th/95th : 25.9 / 28.6 bar

mc_result.plot_histogram(bins=40, title="LOX Pump Outlet Pressure Distribution")
mc_result.save("outputs/pump_mc_1000.hdf5")

# ---------------------------------------------------------------------------
# 3. Performance map sweep (speed × flow)
# ---------------------------------------------------------------------------
sweep = ComponentSweep(
    rig=rig,
    axes=[
        SweepAxis("omega",      np.linspace(8000, 26000, 20) * np.pi / 30),
        SweepAxis("inlet.mdot", np.linspace(0.4, 2.0, 16)),
    ],
    fixed_bcs={"inlet.P": P_inlet, "inlet.h": h_inlet},
    outputs=["outlet.P", "eta", "power"],
    n_jobs=-1,
)
map_result = sweep.run()
print(f"Sweep complete. Failed points: {map_result.n_failed}")

map_result.plot_map(
    "eta",
    x_axis="inlet.mdot", y_axis="omega",
    x_label="Flow rate [kg/s]",
    y_label="Shaft speed [rpm]",
    y_scale=30 / np.pi,
    colorbar_label="Pump efficiency",
    title="LOX Pump Efficiency Map",
)
map_result.plot_map(
    "outlet.P",
    x_axis="inlet.mdot", y_axis="omega",
    x_label="Flow rate [kg/s]",
    y_label="Shaft speed [rpm]",
    y_scale=30 / np.pi,
    colorbar_label="Outlet pressure [bar]",
    title="LOX Pump Head Map",
)
map_result.save("outputs/pump_map_sweep.hdf5")
