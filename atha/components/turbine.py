# atha/components/turbine.py
from __future__ import annotations
import math
from typing import Dict, Optional
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, ShaftPort, PortDirection


class TurbineMap:
    """
    Design-point based turbine performance map.

    Efficiency is modelled as constant at η_design for simplicity.
    Extend with a (PR, u/c₀) table for higher fidelity.
    """

    def __init__(
        self,
        PR_design: float,
        eta_design: float,
        mdot_corrected_design: float,
    ) -> None:
        self._PR_design = PR_design
        self._eta_design = eta_design
        self._mdot_corrected_design = mdot_corrected_design

    @classmethod
    def from_design_point(
        cls,
        PR_design: float,
        eta_design: float,
        mdot_corrected_design: float,
    ) -> "TurbineMap":
        return cls(PR_design, eta_design, mdot_corrected_design)

    def efficiency(self, PR: float = 0.0, mdot_corrected: float = 0.0) -> float:
        return self._eta_design

    @property
    def PR_design(self) -> float:
        return self._PR_design


class Turbine(BaseComponent):
    """
    Algebraic turbine with TurbineMap performance model.

    Uses ideal-gas isentropic expansion to compute enthalpy drop:
        delta_h_s = h_in * (1 - PR^(-(gamma-1)/gamma))   [ideal gas approx]
        delta_h   = eta * delta_h_s
        W         = mdot * delta_h
        tau       = W / max(omega, 1.0)

    The hot-gas gamma defaults to 1.3 (typical for GG exhaust).
    Override via ``inputs["inlet.gamma"]`` if the upstream component
    provides it.

    Ports
    -----
    inlet   FluidPort INLET
    outlet  FluidPort OUTLET
    shaft   ShaftPort OUTLET  — delivers torque to shaft
    """

    def __init__(
        self,
        name: str,
        diameter: float,
        turbine_map: TurbineMap,
        gamma: float = 1.3,
        efficiency_map=None,
    ) -> None:
        self._diameter = diameter
        self._turbine_map = turbine_map
        self._gamma = gamma
        self._efficiency_map = efficiency_map
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("inlet",  FluidPort("inlet",  PortDirection.INLET,  self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))
        self._register_port("shaft",  ShaftPort("shaft",  PortDirection.OUTLET, self))

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
        P_in  = inputs.get("inlet.P",  1e6)
        h_in  = inputs.get("inlet.h",  1e6)
        P_out = inputs.get("outlet.P", 1e5)
        mdot  = inputs.get("inlet.mdot", 1.0)
        omega = inputs.get("shaft.omega", 1000.0)
        gamma = inputs.get("inlet.gamma", self._gamma)

        PR = P_in / max(P_out, 1.0)

        if self._efficiency_map is not None:
            ctx = dict(states)
            ctx.update(inputs)
            ctx["PR"] = PR
            eta = self._efficiency_map.evaluate(ctx)["efficiency"]
        else:
            eta = self._turbine_map.efficiency(PR)

        # Ideal-gas isentropic enthalpy drop: Δh_s = h_in*(1 - PR^(-(γ-1)/γ))
        exp = (gamma - 1.0) / gamma
        delta_h_s = max(h_in * (1.0 - PR ** (-exp)), 0.0)
        delta_h   = eta * delta_h_s
        h_out     = h_in - delta_h
        W         = abs(mdot) * delta_h
        tau       = W / max(abs(omega), 1.0)

        return {
            "outlet.P":   P_out,
            "outlet.h":   h_out,
            "delta_h":    delta_h,
            "power":      W,
            "tau_drive":  tau,
            "efficiency": eta,
        }

    def get_state_derivatives(self, t, states, inputs, outputs):
        return {}

    def get_residuals(self, t, states, inputs, outputs):
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        pass
