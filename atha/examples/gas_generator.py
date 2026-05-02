#!/usr/bin/env python3
"""
Gas generator cycle example.

Simplified topology (no turbomachinery for Phase 9):
  [Fuel Volume] ---> [Combustion Chamber] ---> [Nozzle]
  [Ox Volume]   --->

This demonstrates:
1. Building an engine from components
2. Running steady-state JANNAF performance calculation
3. Printing a performance summary
"""
from __future__ import annotations
from atha.thermo.ideal_gas import IdealGasBackend
from atha.jannaf.simplified import SimplifiedJANNAF, G0
from atha.jannaf.efficiency import JANNAFEfficiencies


def main():
    # Simplified RP-1/LOX gas generator approximation
    # Products: gamma≈1.25, R≈330 J/(kg·K), T≈3400K at MR=2.7
    thermo = IdealGasBackend(gamma=1.25, R=330.0)
    eff = JANNAFEfficiencies(
        eta_cstar=0.97,
        eta_Cd=0.98,
        eta_velocity=0.99,
        eta_divergence=0.983,
        eta_two_phase=1.0,
        eta_boundary_layer=0.99,
    )

    # Engine parameters (Merlin-class, approximate)
    Pc = 9.7e6         # Pa (~97 bar)
    T_chamber = 3400.0 # K
    MR = 2.7           # O/F for RP-1/LOX
    At = 0.092         # m² (throat)
    epsilon = 16.0     # expansion ratio (Merlin sea-level)
    Pa = 101325.0      # Pa (sea level)
    mdot = 277.0       # kg/s (approximate)

    jannaf = SimplifiedJANNAF(thermo, eff,
                               throat_area=At,
                               exit_area=At * epsilon,
                               ambient_pressure=Pa)
    result = jannaf.compute(Pc, T_chamber, MR, mdot)

    print("=" * 55)
    print("  ATHA Gas Generator Cycle — Performance Summary")
    print("=" * 55)
    print(f"  Chamber pressure:    {Pc/1e6:.2f} MPa")
    print(f"  Chamber temperature: {T_chamber:.0f} K")
    print(f"  O/F ratio:           {MR:.2f}")
    print(f"  Throat area:         {At:.4f} m²")
    print(f"  Expansion ratio:     {epsilon:.1f}")
    print(f"  Propellant flow:     {mdot:.1f} kg/s")
    print("-" * 55)
    print(f"  c* ideal:            {result.c_star_ideal:.1f} m/s")
    print(f"  c* delivered:        {result.c_star:.1f} m/s")
    print(f"  Cf (delivered):      {result.Cf:.4f}")
    print(f"  Thrust:              {result.thrust/1e3:.1f} kN")
    print(f"  Isp (delivered):     {result.Isp:.1f} s")
    print(f"  Isp (ideal):         {result.Isp_ideal:.1f} s")
    print(f"  Exit pressure:       {result.P_exit/1e3:.2f} kPa")
    print("=" * 55)


if __name__ == "__main__":
    main()
