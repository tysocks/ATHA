"""Simple TCA PMS target-profile example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atha.config import load_analysis_config
from atha.examples.tca import run_tca_profile


CONFIG_PATH = Path(__file__).parent / "configs" / "analysis.yaml"


def main():
    loaded = load_analysis_config(CONFIG_PATH)
    result = run_tca_profile(CONFIG_PATH)
    source = loaded.operating_conditions.targets["pms"]["schedule"]["source"]
    source_label = source.get("path") if isinstance(source, dict) else source

    print("\nTCA PMS target profile")
    print(f"  Target source   : {source_label}")
    print(f"  mdot_total box  : {result.mdot_total.min():.3f}-{result.mdot_total.max():.3f} kg/s")
    print(f"  OF box          : {result.of_ratio.min():.3f}-{result.of_ratio.max():.3f}")
    print(f"  Pc range        : {result.pc.min() / 1e6:.3f}-{result.pc.max() / 1e6:.3f} MPa")
    print(f"  Thrust range    : {np.nanmin(result.thrust) / 1000:.3f}-{np.nanmax(result.thrust) / 1000:.3f} kN")
    print(f"  CSV             : {result.csv}")
    print(f"  Plot            : {result.plot}")


if __name__ == "__main__":
    main()
