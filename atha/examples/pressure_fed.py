#!/usr/bin/env python3
"""
Pressure-fed engine example with transient simulation.

Topology:
  [Volume (tank, high P)] ---> [Combustion Chamber (simplified)] ---> [Nozzle]

Demonstrates transient simulation: pressure vessel discharging through a nozzle.
"""
from __future__ import annotations
import numpy as np
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
from atha.solver.transient import TransientSolver


def main():
    gas = IdealGasBackend(gamma=1.4, R=287.0)

    # Large pressurized tank: 10 m³, 5 MPa, 400 K
    tank = Volume("tank", volume=10.0, thermo=gas,
                  initial_P=5e6, initial_T=400.0)
    tank.add_outlet("outlet")

    engine = Engine("pressure_fed")
    engine.add_component(tank)
    layout = engine.compile()

    X0 = layout.assemble_state_vector()
    fs0 = gas.state_from_PT(5e6, 400.0)

    # Constant mass outflow (nozzle choked, so approximately constant for high PR)
    mdot_out = 10.0  # kg/s

    def bcs(t):
        return {"outlet.mdot": mdot_out}

    solver = TransientSolver(layout, method="Radau",
                             rtol=1e-4, atol=1e-6, max_step=0.5)
    result = solver.integrate((0.0, 50.0), X0, bcs)

    P = result.get("tank", "P")
    t = result.t

    # Initial tank mass: rho * V
    rho0 = fs0.rho
    m0 = rho0 * 10.0

    print("=" * 55)
    print("  ATHA Pressure-Fed Engine — Transient Discharge")
    print("=" * 55)
    print(f"  Initial tank pressure: {P[0]/1e6:.2f} MPa")
    print(f"  Initial tank mass:     {m0:.1f} kg")
    print(f"  Outflow rate:          {mdot_out:.1f} kg/s")
    print(f"  Simulation time:       {t[-1]:.0f} s")
    print("-" * 55)
    print(f"  Final tank pressure:   {P[-1]/1e6:.2f} MPa")
    print(f"  Pressure drop:         {(P[0]-P[-1])/1e6:.2f} MPa")
    print(f"  Time steps recorded:   {len(t)}")
    print("=" * 55)

    # Verify pressure dropped
    assert P[-1] < P[0], "Tank pressure should decrease during discharge"
    print("  [PASS] Tank pressure decreased as expected")


if __name__ == "__main__":
    main()
