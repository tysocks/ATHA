"""
Gas Generator Cycle — Single Turbopump Shaft
Monte Carlo Uncertainty Study + Speed Sweep
============================================
Engine topology:

  LOX tank ──► LOX pump ──┬──► Main LOX injector ──► Combustion chamber ──► Nozzle
                           │                                ▲
  Fuel tank ──► Fuel pump ─┼──► Main fuel injector ────────┘
                           │
                           └──► Gas generator (small fraction, ~3–4% of total flow)
                                     │
                                     ▼
                               GG Turbine (single shaft)
                                     │
                               (exhaust dump overboard)

Both pumps are on the same shaft driven by a single turbine. The gas generator
burns a small propellant bleed at a fuel-rich mixture ratio to produce hot gas
at moderate temperature (~1000 K) safe for the turbine.

Three analyses in this script:
  1. Nominal steady-state balance
  2. Monte Carlo — uncertain pump maps, GG efficiency, propellant temperatures
  3. Speed sweep — vary turbopump speed (pump ``shaft.omega`` BCs) for throttle envelope

NOTE: Requires planned components — Pump, Turbine, Rotor, GasGenerator,
      CombustionChamber, Nozzle, OrificeCompressible.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from atha.core.engine import Engine
from atha.components.pump import Pump, PumpMap
from atha.components.turbine import Turbine, TurbineMap
from atha.components.rotor import Rotor
from atha.components.gas_generator import GasGenerator
from atha.components.combustion_chamber import CombustionChamber
from atha.components.nozzle import Nozzle
from atha.components.orifice import OrificeCompressible
from atha.thermo.coolprop_backend import CoolPropBackend
from atha.thermo.cantera_backend import CanteraBackend
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.solver.steady_state import SteadyStateSolver
from atha.monte_carlo import MonteCarloRunner, UncertainParameter, ParameterType

# ---------------------------------------------------------------------------
# Propellants
# ---------------------------------------------------------------------------
lox     = CoolPropBackend("Oxygen")
ethanol = CoolPropBackend("Ethanol")

P_lox_tank  = 4e5;  T_lox  = 91.0
P_fuel_tank = 4e5;  T_fuel = 293.0

h_lox_inlet  = lox.state_from_PT(P_lox_tank,  T_lox).h
h_fuel_inlet = ethanol.state_from_PT(P_fuel_tank, T_fuel).h

# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------
Pc_design  = 2.0e6    # Pa
MR_main    = 2.8      # LOX/ethanol
At         = 1.963e-3  # m²  (50 mm throat)
Ae_At      = 8.0
F_design   = 5500.0   # N  vacuum

mdot_total = F_design / (290.0 * 9.80665)   # Isp_vac ~290s for GG cycle
mdot_lox   = mdot_total * MR_main / (1 + MR_main)
mdot_fuel  = mdot_total / (1 + MR_main)

# GG bleed: 3.5% of each propellant (fuel-rich MR ~0.4)
GG_BLEED_FRAC = 0.035
MR_gg         = 0.40
mdot_gg_fuel  = GG_BLEED_FRAC * mdot_fuel
mdot_gg_lox   = mdot_gg_fuel * MR_gg

# Pump outlet pressure
P_pump_out = Pc_design * 1.55   # Pa

# ---------------------------------------------------------------------------
# Performance maps
# ---------------------------------------------------------------------------
lox_map  = PumpMap.from_design_point(mdot_lox,  P_pump_out - P_lox_tank,  22000, 0.68)
fuel_map = PumpMap.from_design_point(mdot_fuel, P_pump_out - P_fuel_tank, 22000, 0.65)
turb_map = TurbineMap.from_design_point(PR_design=2.5, eta_design=0.70, mdot_corrected_design=0.05)

# ---------------------------------------------------------------------------
# Build engine factory function
# (called with perturbed parameters for Monte Carlo)
# ---------------------------------------------------------------------------
def build_engine(lox_eta_scale=1.0, fuel_eta_scale=1.0, gg_efficiency=0.93,
                 T_lox_inlet=91.0, T_fuel_inlet=293.0):
    engine = Engine("gg_single")

    shaft = Rotor("shaft", moment_of_inertia=0.10, initial_speed_rpm=22000)

    lox_map_scaled  = lox_map.scale_efficiency(lox_eta_scale)
    fuel_map_scaled = fuel_map.scale_efficiency(fuel_eta_scale)

    lox_pump  = Pump("lox_pump",  diameter=0.09, pump_map=lox_map_scaled,  fluid=lox)
    fuel_pump = Pump("fuel_pump", diameter=0.07, pump_map=fuel_map_scaled, fluid=ethanol)

    gg_thermo = CanteraBackend("gri30.yaml")
    gg = GasGenerator(
        "gg",
        volume=1e-4,
        thermo=gg_thermo,
        fuel="C2H5OH", oxidizer="O2",
        design_MR=MR_gg,
        efficiency=gg_efficiency,
        initial_P=P_pump_out,
        initial_T=1050.0,
    )

    turbine = Turbine("turbine", diameter=0.10, turbine_map=turb_map)

    cc_thermo = CanteraBackend("gri30.yaml")
    chamber = CombustionChamber(
        "chamber", volume=5e-4,
        thermo=cc_thermo,
        fuel="C2H5OH", oxidizer="O2",
        efficiency=0.96,
        initial_P=Pc_design, initial_T=3400.0,
    )

    nozzle_eff = JANNAFEfficiencies(eta_cstar=0.975, eta_divergence=0.983)
    nozzle = Nozzle("nozzle", throat_area=At, exit_area=At * Ae_At, efficiencies=nozzle_eff)

    lox_inj  = OrificeCompressible("lox_inj",  Cd=0.72, area=1.5e-4)
    fuel_inj = OrificeCompressible("fuel_inj", Cd=0.72, area=1.2e-4)

    for comp in [shaft, lox_pump, fuel_pump, gg, turbine,
                 chamber, nozzle, lox_inj, fuel_inj]:
        engine.add_component(comp)

    # Turbopump shaft
    engine.connect(turbine.port("shaft"),   shaft.port("turbine_in"))
    engine.connect(lox_pump.port("shaft"),  shaft.port("pump_lox"))
    engine.connect(fuel_pump.port("shaft"), shaft.port("pump_fuel"))

    # GG propellant bleed
    engine.connect(lox_pump.port("bleed"),  gg.port("lox_inlet"))
    engine.connect(fuel_pump.port("bleed"), gg.port("fuel_inlet"))
    engine.connect(gg.port("outlet"),       turbine.port("inlet"))
    # Turbine exhaust dumps overboard (open port, no downstream component)

    # Main propellant to chamber
    engine.connect(lox_pump.port("outlet"),  lox_inj.port("inlet"))
    engine.connect(fuel_pump.port("outlet"), fuel_inj.port("inlet"))
    engine.connect(lox_inj.port("outlet"),   chamber.port("lox_inlet"))
    engine.connect(fuel_inj.port("outlet"),  chamber.port("fuel_inlet"))
    engine.connect(chamber.port("outlet"),   nozzle.port("inlet"))

    layout = engine.compile()
    return layout, engine


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # 1. Nominal steady-state
    # ---------------------------------------------------------------------------
    OMEGA_DESIGN = 22000 * np.pi / 30.0  # rad/s

    layout, eng = build_engine()
    X0  = layout.assemble_state_vector()
    bcs = {
        "lox_pump.inlet.P":  P_lox_tank,  "lox_pump.inlet.h":  h_lox_inlet,
        "fuel_pump.inlet.P": P_fuel_tank, "fuel_pump.inlet.h": h_fuel_inlet,
        "nozzle.P_ambient":  0.0,
        # Pin shaft to design speed; full torque balance requires a turbine
        # flow model with choked-nozzle capacity — not yet implemented.
        "shaft.omega_override": OMEGA_DESIGN,
    }

    solver = SteadyStateSolver(layout, tol=1e-8)
    X_ss   = solver.solve(X0, bcs)
    layout.scatter_state_vector(X_ss)

    chamber_comp = eng["chamber"]
    shaft_comp   = eng["shaft"]
    nozzle_comp  = eng["nozzle"]

    print(f"\nGG Single-Shaft Steady-State Balance:")
    print(f"  Chamber pressure : {chamber_comp._state_values['P']/1e6:.3f} MPa")
    print(f"  GG temperature   : {eng['gg']._state_values['T']:.0f} K")
    print(f"  Shaft speed      : {shaft_comp._state_values['omega'] * 30/np.pi:.0f} rpm")
    print(f"  Vacuum thrust    : {nozzle_comp.last_outputs['thrust']:.0f} N")
    print(f"  Vacuum Isp       : {nozzle_comp.last_outputs['Isp_vacuum']:.1f} s")

    # ---------------------------------------------------------------------------
    # 2. Monte Carlo — hardware variability
    # ---------------------------------------------------------------------------
    params = [
        UncertainParameter("lox_eta_scale",  ParameterType.NORMAL,  mean=1.0,  std=0.025),
        UncertainParameter("fuel_eta_scale", ParameterType.NORMAL,  mean=1.0,  std=0.025),
        UncertainParameter("gg_efficiency",  ParameterType.NORMAL,  mean=0.93, std=0.02),
        UncertainParameter("T_lox_inlet",    ParameterType.NORMAL,  mean=91.0, std=0.8),
        UncertainParameter("T_fuel_inlet",   ParameterType.NORMAL,  mean=293.0, std=3.0),
    ]

    def evaluate_engine(X: dict) -> float:
        """Return vacuum thrust [N] for one MC sample."""
        h_lox  = lox.state_from_PT(P_lox_tank,  X["T_lox_inlet"]).h
        h_fuel = ethanol.state_from_PT(P_fuel_tank, X["T_fuel_inlet"]).h

        lay, e = build_engine(
            lox_eta_scale  = X["lox_eta_scale"],
            fuel_eta_scale = X["fuel_eta_scale"],
            gg_efficiency  = X["gg_efficiency"],
            T_lox_inlet    = X["T_lox_inlet"],
            T_fuel_inlet   = X["T_fuel_inlet"],
        )
        bcs_mc = {
            "lox_pump.inlet.P":  P_lox_tank,  "lox_pump.inlet.h":  h_lox,
            "fuel_pump.inlet.P": P_fuel_tank, "fuel_pump.inlet.h": h_fuel,
            "nozzle.P_ambient":  0.0,
            "shaft.omega_override": OMEGA_DESIGN,
        }
        X_sol = SteadyStateSolver(lay).solve(lay.assemble_state_vector(), bcs_mc)
        lay.scatter_state_vector(X_sol)
        return e["nozzle"].last_outputs["thrust"]

    _mc_names = [p.name for p in params]

    def evaluate_engine_row(X: np.ndarray) -> float:
        d = {_mc_names[i]: float(X[i]) for i in range(len(_mc_names))}
        return evaluate_engine(d)

    runner = MonteCarloRunner(
        params=params,
        evaluate_fn=evaluate_engine_row,
        n_samples=500,
        sampler="lhs",
        n_jobs=-1,
        seed=7,
    )
    mc_result = runner.run()
    mc_result.print_summary()
    mc_result.plot_histogram(bins=35, title="GG Cycle Vacuum Thrust Distribution")
    mc_result.save("outputs/gg_thrust_mc_500.hdf5")

    # ---------------------------------------------------------------------------
    # 3. Speed sweep — throttle envelope via turbopump shaft speed
    #
    # ``shaft.omega_override`` drives the rotor state to the target speed via a
    # stiff spring in the ODE (see Rotor.get_state_derivatives). The solver
    # propagates the resulting shaft omega to pumps and turbine automatically
    # through port connections, varying pump discharge pressure and chamber
    # pressure with speed.
    # ---------------------------------------------------------------------------

    def evaluate_at_speed(X: dict) -> dict:
        lay, e = build_engine()
        omega = float(X["omega"])
        bcs_sw = dict(bcs)
        bcs_sw["shaft.omega_override"] = omega
        X_sol = SteadyStateSolver(lay).solve(lay.assemble_state_vector(), bcs_sw)
        lay.scatter_state_vector(X_sol)
        return {
            "thrust": e["nozzle"].last_outputs["thrust"],
            "Pc":     e["chamber"]._state_values["P"],
            "Isp":    e["nozzle"].last_outputs["Isp_vacuum"],
            "P_lox_pump_out": e["lox_pump"].last_outputs["outlet.P"],
        }

    speeds_rpm = np.linspace(12000, 26000, 15)
    thrust_arr = np.zeros(len(speeds_rpm))
    Pc_arr     = np.zeros(len(speeds_rpm))
    Isp_arr    = np.zeros(len(speeds_rpm))
    P_lox_out  = np.zeros(len(speeds_rpm))

    for i, rpm in enumerate(speeds_rpm):
        out = evaluate_at_speed({"omega": rpm * np.pi / 30})
        thrust_arr[i] = out["thrust"]
        Pc_arr[i]     = out["Pc"]
        Isp_arr[i]    = out["Isp"]
        P_lox_out[i]  = out["P_lox_pump_out"]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(speeds_rpm, thrust_arr / 1000)
    axes[0].set(xlabel="Pump speed [rpm]", ylabel="Thrust [kN]", title="Thrust vs. pump speed")
    ax1 = axes[1]
    ax1.plot(speeds_rpm, Pc_arr / 1e6, label="Pc (chamber state)")
    ax1.set(xlabel="Pump speed [rpm]", ylabel="Pc [MPa]")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1b = ax1.twinx()
    ax1b.plot(speeds_rpm, P_lox_out / 1e5, color="C1", label="LOX pump outlet")
    ax1b.set(ylabel="LOX pump outlet [bar]")
    ax1b.tick_params(axis="y", labelcolor="C1")
    ax1.set_title("Pc vs. pump speed (outlet P responds)")
    axes[2].plot(speeds_rpm, Isp_arr)
    axes[2].set(xlabel="Pump speed [rpm]", ylabel="Isp [s]", title="Isp vs. pump speed")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/gg_speed_sweep.png", dpi=150)
    plt.close()
    print("\nSpeed sweep range:")
    print(f"  Thrust: {thrust_arr.min()/1000:.2f} – {thrust_arr.max()/1000:.2f} kN")
    print(f"  Pc:     {Pc_arr.min()/1e6:.3f} – {Pc_arr.max()/1e6:.3f} MPa")
    print(f"  LOX pump outlet: {P_lox_out.min()/1e5:.2f} – {P_lox_out.max()/1e5:.2f} bar")
    print(f"  Isp:    {Isp_arr.min():.1f} – {Isp_arr.max():.1f} s")
