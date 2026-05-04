# atha/components/valve.py
from __future__ import annotations
import math
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection


class Valve(BaseComponent):
    """
    Variable-area incompressible valve.

    Physics:
        mdot = Cv * A_frac * A_max * sqrt(2 * rho * |dP|) * sign(dP)

    Ports:
        inlet  (FluidPort, INLET)
        outlet (FluidPort, OUTLET)

    Parameters:
        max_area        [m^2]
        discharge_coeff [-]   default 0.98

    Optional PerformanceMap overrides
    ----------------------------------
    cd_map : PerformanceMap
        Replaces the scalar ``discharge_coeff``.  Must output ``"Cd"``.
        Context axes available: "inlet.P", "outlet.P", "inlet.rho",
        "valve.A_frac", and any other BCS keys.

    cda_map : PerformanceMap
        Provides the effective Cd*A product directly, bypassing both the
        scalar ``discharge_coeff`` and ``max_area``.  Must output ``"CdA"``.
        ``valve.A_frac`` is NOT applied when ``cda_map`` is used — the map
        is expected to encode area-fraction dependence in its axes.
        Takes precedence over ``cd_map``.
    """

    def __init__(
        self,
        name: str,
        max_area: float,
        discharge_coeff: float = 0.98,
        cd_map=None,
        cda_map=None,
    ) -> None:
        self._max_area = max_area
        self._discharge_coeff = discharge_coeff
        self._cd_map = cd_map
        self._cda_map = cda_map
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("inlet", FluidPort("inlet", PortDirection.INLET, self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))

    def _declare_states(self) -> None:
        pass

    def _declare_algebraic_vars(self) -> None:
        pass

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        P_in = inputs.get("inlet.P", 0.0)
        P_out = inputs.get("outlet.P", 0.0)
        rho = inputs.get("inlet.rho", 1.0)
        A_frac = inputs.get("valve.A_frac", 1.0)

        context: Dict[str, float] = dict(states)
        context.update(inputs)

        if self._cda_map is not None:
            CdA = self._cda_map.evaluate(context)["CdA"]
        elif self._cd_map is not None:
            Cd = self._cd_map.evaluate(context)["Cd"]
            CdA = Cd * A_frac * self._max_area
        else:
            CdA = self._discharge_coeff * A_frac * self._max_area

        dP = P_in - P_out
        if dP == 0.0:
            mdot = 0.0
        else:
            mdot = CdA * math.sqrt(2.0 * rho * abs(dP)) * math.copysign(1.0, dP)

        return {"mdot": mdot, "A_frac": A_frac}

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {}

    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        pass
