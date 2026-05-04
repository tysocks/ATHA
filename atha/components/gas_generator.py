# atha/components/gas_generator.py
from __future__ import annotations
from atha.components.preburner import Preburner


class GasGenerator(Preburner):
    """
    Gas generator for turbine drive in GG cycles.

    Same physics as Preburner / CombustionChamber.  Ports:
        lox_inlet   FluidPort INLET   (LOX bleed from pump)
        fuel_inlet  FluidPort INLET   (fuel bleed from pump)
        outlet      FluidPort OUTLET  (hot gas to turbine)
    """
    pass
