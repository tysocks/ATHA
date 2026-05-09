"""Two fixed-supply valve transient chain example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs"


def main():
    result = run_config_folder(CONFIG_PATH).require_summary()
    print("\nTwo-valve transient chain")
    print(f"  Valve A actual  : {result.valve_a_position.min():.3f}-{result.valve_a_position.max():.3f}")
    print(f"  Valve B actual  : {result.valve_b_position.min():.3f}-{result.valve_b_position.max():.3f}")
    print(f"  mdot A          : {result.mdot_a.min():.4f}-{result.mdot_a.max():.4f} kg/s")
    print(f"  mdot B          : {result.mdot_b.min():.4f}-{result.mdot_b.max():.4f} kg/s")
    print(f"  Pc range        : {result.chamber_pressure.min() / 1e5:.3f}-{result.chamber_pressure.max() / 1e5:.3f} bar")
    print(f"  Thrust range    : {np.nanmin(result.thrust):.2f}-{np.nanmax(result.thrust):.2f} N")
    print(f"  CSV             : {result.csv}")
    print(f"  Plot            : {result.plot}")


if __name__ == "__main__":
    main()
