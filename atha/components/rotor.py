# atha/components/rotor.py
from __future__ import annotations
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import ShaftPort, PortDirection


class Rotor(BaseComponent):
    """
    Rotating shaft/rotor with angular momentum conservation.

    State: omega [rad/s]

    Physics:
        dω/dt = (τ_drive - τ_load - k_f * ω) / I

    Ports:
        shaft_in  (ShaftPort, INLET)  — receives drive torque τ_drive
        shaft_out (ShaftPort, OUTLET) — provides angular velocity ω to loads,
                                        reads load torque τ_load
    """

    def __init__(
        self,
        name: str,
        inertia: float,
        friction_coeff: float = 0.0,
        initial_omega: float = 0.0,
    ) -> None:
        self._inertia = inertia
        self._friction_coeff = friction_coeff
        self._initial_omega = initial_omega
        super().__init__(name)
        # Initialize state to initial_omega
        self._state_values["omega"] = initial_omega

    # ── BaseComponent hooks ─────────────────────────────────────────────────

    def _declare_ports(self) -> None:
        self._register_port("shaft_in", ShaftPort("shaft_in", PortDirection.INLET, self))
        self._register_port("shaft_out", ShaftPort("shaft_out", PortDirection.OUTLET, self))

    def _declare_states(self) -> None:
        self._register_state("omega", self._initial_omega)

    def _declare_algebraic_vars(self) -> None:
        pass

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {"omega": states["omega"]}

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        tau_drive = inputs.get("shaft_in.tau", 0.0)
        tau_load = inputs.get("shaft_out.tau", 0.0)
        omega = states["omega"]
        domega_dt = (tau_drive - tau_load - self._friction_coeff * omega) / self._inertia
        return {"omega": domega_dt}

    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        self._state_values["omega"] = operating_point.get("omega", self._initial_omega)
