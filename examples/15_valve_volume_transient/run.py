"""Timed valve opening into a downstream volume."""

from __future__ import annotations

from pathlib import Path

from atha.examples.valve_volume import run_valve_volume_profile


CONFIG_PATH = Path(__file__).parent / "configs" / "analysis.yaml"


def main():
    result = run_valve_volume_profile(CONFIG_PATH)
    print("\nValve-volume transient")
    print(f"  Valve actual    : {result.valve_position.min():.3f}-{result.valve_position.max():.3f}")
    print(f"  Pressure range  : {result.pressure.min() / 1e5:.3f}-{result.pressure.max() / 1e5:.3f} bar")
    print(f"  Inlet mdot      : {result.mdot_in.min():.4f}-{result.mdot_in.max():.4f} kg/s")
    print(f"  Outlet mdot     : {result.mdot_out.min():.4f}-{result.mdot_out.max():.4f} kg/s")
    print(f"  CSV             : {result.csv}")
    print(f"  Plot            : {result.plot}")


if __name__ == "__main__":
    main()
