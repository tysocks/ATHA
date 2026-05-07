# atha/components/gas_generator.py
from __future__ import annotations
from typing import Dict
from atha.components.preburner import Preburner


class GasGenerator(Preburner):
    """
    Gas generator for turbine drive in GG cycles.

    Same physics as Preburner / CombustionChamber.  Ports:
        lox_inlet   FluidPort INLET   (LOX bleed from pump)
        fuel_inlet  FluidPort INLET   (fuel bleed from pump)
        outlet      FluidPort OUTLET  (hot gas to turbine)

    Steady-state residuals use pressure-tracking (P ← upstream pump outlet)
    rather than mass-balance, because the GG has no inlet orifice model and
    the pump outlet pressure sets the GG pressure directly.
    """

    def get_steady_state_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        P = states.get("P", self._initial_P)
        # GG pressure tracks upstream pump outlet pressure
        P_up = inputs.get("lox_inlet.P", inputs.get("ox_inlet.P", self._initial_P))
        r_P = (P_up - P) / max(abs(P_up), 1.0)

        # GG temperature converges to adiabatic flame temperature × efficiency
        T_cur = outputs.get("T", self._initial_T)
        T_tgt = self._eta_cstar * self._T_adiabatic
        r_h = (T_tgt - T_cur) / max(T_tgt, 1.0)

        return {"P": r_P, "h": r_h}
