"""Methane/LOX TCA with LOX valve mass-flow control."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs"


def main():
    result = run_config_folder(CONFIG_PATH).require_summary()
    mdot_total = result.mdot_methane + result.mdot_lox
    print("\nMethane/LOX TCA total mass-flow control")
    print(f"  Methane valve   : {result.methane_valve_position.min():.3f}-{result.methane_valve_position.max():.3f}")
    print(f"  LOX valve       : {result.lox_valve_position.min():.3f}-{result.lox_valve_position.max():.3f}")
    print(f"  Total mdot      : {mdot_total.min():.3f}-{mdot_total.max():.3f} kg/s")
    print(f"  OF range        : {np.nanmin(result.of_ratio):.3f}-{np.nanmax(result.of_ratio):.3f}")
    print(f"  Pc range        : {result.chamber_pressure.min() / 1e5:.3f}-{result.chamber_pressure.max() / 1e5:.3f} bar")
    print(f"  Thrust range    : {np.nanmin(result.thrust):.2f}-{np.nanmax(result.thrust):.2f} N")
    print(f"  CSV             : {result.csv}")
    print(f"  Plot            : {result.plot}")


if __name__ == "__main__":
    main()
