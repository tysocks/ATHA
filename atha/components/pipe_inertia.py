# atha/components/pipe_inertia.py
from __future__ import annotations
import math
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection


class PipeWithInertia(BaseComponent):
    """
    Pipe with flow momentum conservation (dynamic mass flow).

    State: mdot [kg/s]

    Physics:
        dṁ/dt = (A / L) * (P_in - P_out - ΔP_friction)
        ΔP_friction = f * (L/D) * rho * v * |v| / 2   (sign-preserving Darcy-Weisbach)

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
        initial_mdot: float = 0.0,
    ) -> None:
        self._length = length
        self._diameter = diameter
        self._friction_factor = friction_factor
        self._initial_mdot = initial_mdot
        self._area = math.pi * (diameter / 2.0) ** 2
        super().__init__(name)
        self._state_values["mdot"] = initial_mdot

    # ── BaseComponent hooks ─────────────────────────────────────────────────

    def _declare_ports(self) -> None:
        self._register_port("inlet", FluidPort("inlet", PortDirection.INLET, self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))

    def _declare_states(self) -> None:
        self._register_state("mdot", self._initial_mdot)

    def _declare_algebraic_vars(self) -> None:
        pass

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        rho = inputs.get("inlet.rho", 1.0)
        mdot = states["mdot"]
        velocity = mdot / (rho * self._area)
        return {"velocity": velocity}

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        P_in = inputs.get("inlet.P", 0.0)
        P_out = inputs.get("outlet.P", 0.0)
        rho = inputs.get("inlet.rho", 1.0)
        mdot = states["mdot"]

        A = self._area
        v = mdot / (rho * A)

        # Darcy-Weisbach, sign-preserving: f * (L/D) * rho * v * |v| / 2
        dP_friction = (
            self._friction_factor
            * (self._length / self._diameter)
            * rho
            * v
            * abs(v)
            / 2.0
        )

        dmdot_dt = (A / self._length) * (P_in - P_out - dP_friction)
        return {"mdot": dmdot_dt}

    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        self._state_values["mdot"] = operating_point.get("mdot", self._initial_mdot)
