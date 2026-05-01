# atha/components/nozzle.py
from __future__ import annotations
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection
from atha.thermo.interface import ThermoBackend
from atha.jannaf.simplified import SimplifiedJANNAF


class Nozzle(BaseComponent):
    """
    Isentropic nozzle with choked throat and JANNAF efficiency factors.

    Ports:
        inlet  (FluidPort, INLET)   — chamber gas
        outlet (FluidPort, OUTLET)  — exhaust

    Parameters:
        throat_area      [m^2]
        exit_area        [m^2]
        ambient_pressure [Pa]   default 0.0 (vacuum)
        thermo           ThermoBackend
        discharge_coeff  [-]    default 0.98
        eta_velocity     [-]    default 0.99
        eta_divergence   [-]    default 0.983
    """

    def __init__(
        self,
        name: str,
        throat_area: float,
        exit_area: float,
        thermo: ThermoBackend,
        ambient_pressure: float = 0.0,
        discharge_coeff: float = 0.98,
        eta_velocity: float = 0.99,
        eta_divergence: float = 0.983,
    ) -> None:
        self._throat_area = throat_area
        self._exit_area = exit_area
        self._thermo = thermo
        self._ambient_pressure = ambient_pressure
        self._discharge_coeff = discharge_coeff
        self._eta_velocity = eta_velocity
        self._eta_divergence = eta_divergence
        self._epsilon = exit_area / throat_area
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
        P_c = inputs.get("inlet.P", 1e6)
        h_c = inputs.get("inlet.h", 1e6)
        mdot = inputs.get("inlet.mdot", 0.0)

        fs = self._thermo.state_from_Ph(P_c, h_c)
        gamma = fs.gamma

        # Exit Mach from area ratio
        Me = SimplifiedJANNAF._exit_mach(gamma, self._epsilon)

        # Exit pressure
        Pe = P_c * (1.0 + (gamma - 1.0) / 2.0 * Me ** 2) ** (-gamma / (gamma - 1.0))

        # Ideal Cf
        Cf_ideal = SimplifiedJANNAF._ideal_cf(gamma, P_c, Pe, self._ambient_pressure, self._epsilon)

        # Delivered Cf with efficiency factors
        Cf_del = self._eta_velocity * self._eta_divergence * Cf_ideal

        # Thrust
        thrust = Cf_del * P_c * self._throat_area

        # Characteristic velocity
        if mdot > 1e-10:
            c_star = P_c * self._throat_area / mdot
        else:
            c_star = 0.0

        return {
            "thrust": thrust,
            "Cf": Cf_del,
            "P_exit": Pe,
            "c_star": c_star,
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
