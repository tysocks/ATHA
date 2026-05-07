# atha/components/preburner.py
from __future__ import annotations
from typing import Dict
from atha.components.combustion_chamber import CombustionChamber


class Preburner(CombustionChamber):
    """
    Fuel-rich or oxidizer-rich preburner for staged-combustion cycles.

    Identical physics to CombustionChamber.  Steady-state residuals use
    pressure-tracking (P ← upstream pump outlet) rather than mass-balance,
    because the preburner has no inlet orifice model and its pressure is set
    directly by the upstream pump.
    """

    def get_steady_state_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        P = states.get("P", self._initial_P)
        # Preburner pressure tracks upstream LOX pump outlet
        P_up = inputs.get("lox_inlet.P", inputs.get("ox_inlet.P", self._initial_P))
        r_P = (P_up - P) / max(abs(P_up), 1.0)

        # Enthalpy: temperature converges to adiabatic flame temperature × efficiency
        T_cur = outputs.get("T", self._initial_T)
        T_tgt = self._eta_cstar * self._T_adiabatic
        r_h = (T_tgt - T_cur) / max(T_tgt, 1.0)

        return {"P": r_P, "h": r_h}
