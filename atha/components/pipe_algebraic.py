# atha/components/pipe_algebraic.py
from __future__ import annotations
import math
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection


class PipeAlgebraic(BaseComponent):
    """
    Quasi-steady incompressible pipe with algebraic pressure-drop / flow relation.

    No states. Mass flow is computed directly from pressure drop.

    Physics:
        ΔP = K * rho * v^2 / 2   where K = f * L/D
        v  = sign(ΔP) * sqrt(2 * |ΔP| / (K * rho))
        ṁ  = rho * A * v

    Ports:
        inlet  (FluidPort, INLET)
        outlet (FluidPort, OUTLET)
    """

    def __init__(
        self,
        name: str,
        length: float,
        diameter: float,
        friction_factor: float = 0.02,
    ) -> None:
        self._length = length
        self._diameter = diameter
        self._friction_factor = friction_factor
        self._area = math.pi * (diameter / 2.0) ** 2
        super().__init__(name)

    # ── BaseComponent hooks ─────────────────────────────────────────────────

    def _declare_ports(self) -> None:
        self._register_port("inlet", FluidPort("inlet", PortDirection.INLET, self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))

    def _declare_states(self) -> None:
        pass  # no states

    def _declare_algebraic_vars(self) -> None:
        pass  # algebraic var computed directly, not via residual

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        P_in = inputs.get("inlet.P", 0.0)
        P_out = inputs.get("outlet.P", 0.0)
        rho = inputs.get("inlet.rho", 1.0)

        dP = P_in - P_out
        A = self._area
        # K = f * L/D, guard against zero
        K = max(self._friction_factor * self._length / self._diameter, 1e-10)

        if abs(dP) < 1e-20:
            v = 0.0
        else:
            sign_dP = 1.0 if dP > 0.0 else -1.0
            v = sign_dP * math.sqrt(2.0 * abs(dP) / (K * rho))

        mdot = rho * A * v
        return {"mdot": mdot, "velocity": v}

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
