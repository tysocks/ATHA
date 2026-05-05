# atha/components/combustion_chamber.py
from __future__ import annotations
from typing import Dict, Optional
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection
from atha.thermo.interface import ThermoBackend


class CombustionChamber(BaseComponent):
    """
    Lumped combustion chamber volume with combustion heat release.

    States: P [Pa], h [J/kg]

    Ports
    -----
    lox_inlet   FluidPort INLET   (also accessible as ox_inlet)
    fuel_inlet  FluidPort INLET
    outlet      FluidPort OUTLET

    Parameters
    ----------
    volume       m³
    thermo       ThermoBackend
    fuel         str   fuel species name (e.g. "CH4", "C2H5OH")
    oxidizer     str   oxidizer species name (e.g. "O2")
    efficiency   -     combustion efficiency (η_c*), default 0.975
    design_MR    -     design mixture ratio (stored, not used in physics)
    initial_P    Pa    default 1e5
    initial_T    K     default 300.0
    T_adiabatic  K     adiabatic flame temperature; defaults to initial_T
    eta_cstar    -     legacy alias for efficiency
    """

    def __init__(
        self,
        name: str,
        volume: float,
        thermo: ThermoBackend,
        fuel: str = "H2",
        oxidizer: str = "O2",
        efficiency: float = 0.975,
        design_MR: Optional[float] = None,
        initial_P: float = 1e5,
        initial_T: float = 300.0,
        # Legacy params kept for backward compatibility
        T_adiabatic: Optional[float] = None,
        eta_cstar: Optional[float] = None,
    ) -> None:
        self._volume = volume
        self._thermo = thermo
        self._fuel = fuel
        self._oxidizer = oxidizer
        self._eta_cstar = eta_cstar if eta_cstar is not None else efficiency
        self._design_MR = design_MR
        self._initial_P = initial_P
        self._initial_T = initial_T
        # T_adiabatic: use explicit value if given, else use initial_T as proxy
        self._T_adiabatic = T_adiabatic if T_adiabatic is not None else initial_T
        super().__init__(name)

    def _declare_ports(self) -> None:
        lox_p = FluidPort("lox_inlet", PortDirection.INLET, self)
        self._register_port("lox_inlet", lox_p)
        self._register_port("ox_inlet",  lox_p)   # alias
        self._register_port("fuel_inlet", FluidPort("fuel_inlet", PortDirection.INLET,  self))
        self._register_port("outlet",     FluidPort("outlet",     PortDirection.OUTLET, self))

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
        P = max(states.get("P", self._initial_P), 1.0)  # clamp: Cantera rejects P <= 0
        h = states.get("h", self._thermo.state_from_PT(self._initial_P, self._initial_T).h)
        try:
            fs = self._thermo.state_from_Ph(P, h)
        except Exception:
            # Fallback when Newton wanders into unphysical (h, P) during iteration
            fs = self._thermo.state_from_PT(P, self._initial_T)
        # Cache T so examples can read _state_values["T"]
        self._state_values["T"] = fs.T
        return {
            "fluid_state": fs,
            "T": fs.T,
            "P": P,
            "rho": fs.rho,
            "gamma": fs.gamma,
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

        mdot_lox  = inputs.get("lox_inlet.mdot",  inputs.get("ox_inlet.mdot", 0.0))
        mdot_fuel = inputs.get("fuel_inlet.mdot", 0.0)
        mdot_out  = inputs.get("outlet.mdot", 0.0)
        mdot_in   = mdot_lox + mdot_fuel
        mdot_net  = mdot_in - mdot_out

        P = max(states.get("P", self._initial_P), 1.0)
        h = states.get("h", 0.0)

        # When no propellant flows (e.g. isolated component during Newton iteration),
        # return a soft spring toward the design-point state rather than a trivially
        # zero residual.  Trivially zero residuals make P and h unobservable, causing
        # Newton to wander into unphysical territory.
        if mdot_in < 1e-12 and abs(mdot_out) < 1e-12:
            try:
                h0 = self._thermo.state_from_PT(self._initial_P, self._initial_T).h
            except Exception:
                h0 = h
            k_soft = 1e-4
            return {"P": k_soft * (self._initial_P - P),
                    "h": k_soft * (h0 - h)}

        # Combustion: effective enthalpy at adiabatic flame temp * efficiency
        try:
            h_ad = self._thermo.state_from_PT(P, self._T_adiabatic).h
        except Exception:
            h_ad = h
        h_in_flux = mdot_in * self._eta_cstar * h_ad
        h_out_flux = mdot_out * h

        dP_dt = (fs.gamma * R_eff * fs.T / V) * (mdot_net / fs.rho)
        dh_dt = (1.0 / m) * (h_in_flux - h_out_flux - V * dP_dt)

        return {"P": dP_dt, "h": dh_dt}

    def get_residuals(self, t, states, inputs, outputs):
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        P = operating_point.get("P", self._initial_P)
        T = operating_point.get("T", self._initial_T)
        self._state_values["P"] = P
        self._state_values["h"] = self._thermo.state_from_PT(P, T).h
        self._state_values["T"] = T
