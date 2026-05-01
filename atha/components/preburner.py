# atha/components/preburner.py
from __future__ import annotations
from atha.components.combustion_chamber import CombustionChamber


class Preburner(CombustionChamber):
    """
    Fuel-rich or oxidizer-rich preburner.
    Same physics as CombustionChamber, with typically lower T_adiabatic.
    """
    pass
