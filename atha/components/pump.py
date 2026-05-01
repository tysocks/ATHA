# atha/components/pump.py
from __future__ import annotations
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, ShaftPort, PortDirection


class Pump(BaseComponent):
    """
    Algebraic pump with speed-scaled pressure rise.

    Physics (simplified design-point model):
        delta_P = delta_P_design * (omega / omega_design)^2
        W       = mdot * delta_P / (rho * efficiency)
        tau     = W / max(omega, 1.0)   [load on shaft]

    Ports:
        inlet  (FluidPort, INLET)
        outlet (FluidPort, OUTLET)
        shaft  (ShaftPort, INLET)  — receives shaft speed omega

    Parameters:
        delta_P_design  [Pa]
        mdot_design     [kg/s]
        omega_design    [rad/s]
        efficiency      [-]    default 0.75
    """

    def __init__(
        self,
        name: str,
        delta_P_design: float,
        mdot_design: float,
        omega_design: float,
        efficiency: float = 0.75,
    ) -> None:
        self._delta_P_design = delta_P_design
        self._mdot_design = mdot_design
        self._omega_design = omega_design
        self._efficiency = efficiency
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("inlet", FluidPort("inlet", PortDirection.INLET, self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))
        self._register_port("shaft", ShaftPort("shaft", PortDirection.INLET, self))

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
        omega = inputs.get("shaft.omega", self._omega_design)
        rho = inputs.get("inlet.rho", 1000.0)
        mdot = inputs.get("inlet.mdot", self._mdot_design)

        delta_P = self._delta_P_design * (omega / self._omega_design) ** 2
        W = mdot * delta_P / (rho * self._efficiency)
        tau = W / max(omega, 1.0)

        return {"delta_P": delta_P, "power": W, "tau_load": tau}

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
