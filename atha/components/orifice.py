# atha/components/orifice.py
from __future__ import annotations
import math
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection


class OrificeCompressible(BaseComponent):
    """
    Compressible orifice with choked/subsonic flow and smooth blending.

    No states (purely algebraic).

    Physics:
        Critical pressure ratio:
            PR_crit = ((gamma+1)/2)^(gamma/(gamma-1))

        Subsonic (PR < PR_crit):
            ṁ_sub = Cd * A * P_in * sqrt(gamma/(R*T_in))
                    * sqrt(2/(gamma-1))
                    * sqrt((P_out/P_in)^(2/gamma) - (P_out/P_in)^((gamma+1)/gamma))

        Choked (PR >= PR_crit):
            ṁ_choked = Cd * A * P_in * sqrt(gamma/(R*T_in))
                       * (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))

        Blended smoothly using tanh at the transition.

    If P_out >= P_in: mdot = 0.0 (no backflow modeled).

    Optional PerformanceMap overrides
    ----------------------------------
    cd_map : PerformanceMap
        Replaces the scalar ``discharge_coeff``.  Must output ``"Cd"``.
        Context axes available: "inlet.P", "outlet.P", "inlet.T",
        "inlet.mdot", and any other BCS keys.

    cda_map : PerformanceMap
        Provides the effective discharge-area product Cd*A directly,
        bypassing both the scalar ``discharge_coeff`` and ``area``.
        Must output ``"CdA"``.  Takes precedence over ``cd_map``.

    Ports:
        inlet  (FluidPort, INLET)
        outlet (FluidPort, OUTLET)
    """

    def __init__(
        self,
        name: str,
        area: float,
        discharge_coeff: float = 0.98,
        Cd: float = None,          # alias for discharge_coeff
        gamma: float = 1.4,
        R_gas: float = 287.0,
        cd_map=None,
        cda_map=None,
    ) -> None:
        self._area = area
        self._discharge_coeff = Cd if Cd is not None else discharge_coeff
        self._gamma = gamma
        self._R_gas = R_gas
        self._cd_map = cd_map
        self._cda_map = cda_map
        super().__init__(name)

    # ── BaseComponent hooks ─────────────────────────────────────────────────

    def _declare_ports(self) -> None:
        self._register_port("inlet", FluidPort("inlet", PortDirection.INLET, self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))

    def _declare_states(self) -> None:
        pass  # no states

    def _declare_algebraic_vars(self) -> None:
        pass

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        P_in = inputs.get("inlet.P", 1e5)
        P_out = inputs.get("outlet.P", 0.9e5)
        T_in = inputs.get("inlet.T", 300.0)

        # No flow against pressure gradient
        if P_out >= P_in:
            return {"mdot": 0.0}

        gamma = self._gamma
        R = self._R_gas

        context: Dict[str, float] = dict(states)
        context.update(inputs)

        # Resolve effective Cd*A
        if self._cda_map is not None:
            CdA = self._cda_map.evaluate(context)["CdA"]
        elif self._cd_map is not None:
            Cd = self._cd_map.evaluate(context)["Cd"]
            CdA = Cd * self._area
        else:
            CdA = self._discharge_coeff * self._area

        PR = P_in / P_out  # pressure ratio >= 1

        # Critical pressure ratio for choked flow
        PR_crit = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))

        # Common factor (without Cd*A)
        common = CdA * P_in * math.sqrt(gamma / (R * T_in))

        # Choked mass flow
        choked_coeff = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        mdot_choked = common * choked_coeff

        # Subsonic mass flow
        pr_ratio = P_out / P_in
        sub_term = (
            pr_ratio ** (2.0 / gamma)
            - pr_ratio ** ((gamma + 1.0) / gamma)
        )
        if sub_term < 0.0:
            sub_term = 0.0
        mdot_sub = common * math.sqrt(2.0 / (gamma - 1.0)) * math.sqrt(sub_term)

        # Smooth blend: blend=1 → fully choked, blend=0 → fully subsonic
        blend = 0.5 + 0.5 * math.tanh((PR - PR_crit) / (0.1 * PR_crit))
        mdot = (1.0 - blend) * mdot_sub + blend * mdot_choked

        return {"mdot": mdot}

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
        pass  # no-op
