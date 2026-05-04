# atha/components/preburner.py
from __future__ import annotations
from atha.components.combustion_chamber import CombustionChamber


class Preburner(CombustionChamber):
    """
    Fuel-rich or oxidizer-rich preburner.

    Identical physics to CombustionChamber.  The ``design_MR`` parameter is
    stored for reference (e.g. monitoring) but the energy-release model uses
    the same ``T_adiabatic`` / ``initial_T`` proxy as the main chamber.
    """
    pass
