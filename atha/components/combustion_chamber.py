# atha/components/combustion_chamber.py
from __future__ import annotations
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection
from atha.thermo.interface import ThermoBackend


class CombustionChamber(BaseComponent):
    """
    Lumped combustion chamber volume with combustion heat release.

    States: P [Pa], h [J/kg]

    Ports:
        fuel_inlet  (FluidPort, INLET)
        ox_inlet    (FluidPort, INLET)
        outlet      (FluidPort, OUTLET)

    Parameters:
        volume       [m^3]
        thermo       ThermoBackend
        T_adiabatic  [K]    adiabatic flame temperature target
        eta_cstar    [-]    combustion efficiency, default 0.975
        initial_P    [Pa]   default 1e5
        initial_T    [K]    default 300.0
    """

    def __init__(
        self,
        name: str,
        volume: float,
        thermo: ThermoBackend,
        T_adiabatic: float,
        eta_cstar: float = 0.975,
        initial_P: float = 1e5,
        initial_T: float = 300.0,
    ) -> None:
        self._volume = volume
        self._thermo = thermo
        self._T_adiabatic = T_adiabatic
        self._eta_cstar = eta_cstar
        self._initial_P = initial_P
        self._initial_T = initial_T
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("fuel_inlet", FluidPort("fuel_inlet", PortDirection.INLET, self))
        self._register_port("ox_inlet", FluidPort("ox_inlet", PortDirection.INLET, self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))

    def _declare_states(self) -> None:
        self._register_state("P", self._initial_P)
        h0 = self._thermo.state_from_PT(self._initial_P, self._initial_T).h
        self._register_state("h", h0)

    def _declare_algebraic_vars(self) -> None:
        pass

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        fs = self._thermo.state_from_Ph(states["P"], states["h"])
        return {
            "fluid_state": fs,
            "T": fs.T,
            "rho": fs.rho,
        }

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        fs = outputs["fluid_state"]
        V = self._volume
        m = fs.rho * V
        R_eff = fs.cp - fs.cv

        mdot_fuel = inputs.get("fuel_inlet.mdot", 0.0)
        mdot_ox = inputs.get("ox_inlet.mdot", 0.0)
        mdot_out = inputs.get("outlet.mdot", 0.0)
        mdot_total_in = mdot_fuel + mdot_ox
        mdot_net = mdot_total_in - mdot_out

        # Combustion: effective enthalpy = eta_cstar * h at adiabatic flame T
        if mdot_total_in > 1e-12:
            h_combustion = self._eta_cstar * self._thermo.state_from_PT(
                states["P"], self._T_adiabatic
            ).h
            h_in_flux = mdot_total_in * h_combustion
        else:
            h_in_flux = 0.0

        h_out_flux = mdot_out * states["h"]

        # Pressure ODE
        dP_dt = (fs.gamma * R_eff * fs.T / V) * (mdot_net / fs.rho)

        # Enthalpy ODE
        dh_dt = (1.0 / m) * (h_in_flux - h_out_flux - V * dP_dt)

        return {"P": dP_dt, "h": dh_dt}

    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        P = operating_point.get("P", self._initial_P)
        T = operating_point.get("T", self._initial_T)
        self._state_values["P"] = P
        self._state_values["h"] = self._thermo.state_from_PT(P, T).h
