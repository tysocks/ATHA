"""Methalox single-shaft gas-generator transient example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atha.runner import run_config_folder


CONFIG_PATH = Path(__file__).parent / "configs"


def main():
    result = run_config_folder(CONFIG_PATH).require_summary()
    print("\nMethalox gas-generator single-shaft transient")
    print(f"  Components      : {result.component_count}")
    print(f"  Connections     : {result.connection_count}")
    print(f"  Solver status   : {result.solver_status}")
    print(f"  Final mdot      : {result.mdot_total[-1]:.3f} kg/s")
    print(f"  Peak thrust     : {np.nanmax(result.thrust):.1f} N")
    print(f"  Final thrust    : {result.thrust[-1]:.1f} N")
    print(f"  Shaft range     : {np.nanmin(result.shaft_rpm):.0f}-{np.nanmax(result.shaft_rpm):.0f} rpm")
    print(f"  CSV             : {result.csv}")
    print(f"  Plot            : {result.plot}")


if __name__ == "__main__":
    main()
