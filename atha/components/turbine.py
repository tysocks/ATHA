# atha/components/turbine.py
from __future__ import annotations
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, ShaftPort, PortDirection
from atha.thermo.interface import ThermoBackend


class Turbine(BaseComponent):
    """
    Algebraic turbine with isentropic expansion.

    Physics:
        delta_h_s      = h_in - h_out_isentropic
        delta_h_actual = efficiency * delta_h_s
        W              = mdot * delta_h_actual
        tau_drive      = W / max(omega, 1.0)

    Ports:
        inlet  (FluidPort, INLET)
        outlet (FluidPort, OUTLET)
        shaft  (ShaftPort, OUTLET)  — delivers torque to shaft

    Parameters:
        efficiency  [-]              default 0.85
        thermo      ThermoBackend
    """

    def __init__(
        self,
        name: str,
        thermo: ThermoBackend,
        efficiency: float = 0.85,
    ) -> None:
        self._thermo = thermo
        self._efficiency = efficiency
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("inlet", FluidPort("inlet", PortDirection.INLET, self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))
        self._register_port("shaft", ShaftPort("shaft", PortDirection.OUTLET, self))

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
        P_in = inputs.get("inlet.P", 1e6)
        h_in = inputs.get("inlet.h", 1e6)
        P_out = inputs.get("outlet.P", 1e5)
        mdot = inputs.get("inlet.mdot", 1.0)
        omega = inputs.get("shaft.omega", 100.0)

        state_in = self._thermo.state_from_Ph(P_in, h_in)
        state_out_s = self._thermo.isentropic_expansion(state_in, P_out)

        delta_h_s = h_in - state_out_s.h
        delta_h_actual = self._efficiency * delta_h_s
        W = mdot * delta_h_actual
        tau = W / max(omega, 1.0)

        return {
            "delta_h": delta_h_actual,
            "power": W,
            "tau_drive": tau,
            "T_exit": state_out_s.T,
        }

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
