# atha/components/volume.py
from __future__ import annotations
from typing import Dict, List
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, ThermalPort, PortDirection
from atha.thermo.interface import ThermoBackend


class Volume(BaseComponent):
    """
    Lumped fluid volume with mass and energy conservation.

    States: P [Pa], h [J/kg]

    Conservation equations (ROCKETS system formulation with P, h as states):

        dP/dt = (gamma_eff * R_eff * T / V) * (sum_in(mdot) - sum_out(mdot))

        dh/dt = (1/m) * (Q_dot_net
                         + sum_in(mdot * h_in)
                         - sum_out(mdot * h_out)
                         - V * dP/dt)

    where m = rho * V, and gamma_eff, R_eff are from the ThermoBackend.

    Ports are added dynamically via add_inlet() / add_outlet().
    A ThermalPort named 'heat' can optionally receive Q_dot from a MetalNode.
    """

    def __init__(
        self,
        name: str,
        volume: float,               # m^3
        thermo: ThermoBackend,
        initial_P: float = 1e5,      # Pa
        initial_T: float = 300.0,    # K
    ) -> None:
        self._volume = volume
        self._thermo = thermo
        self._initial_P = initial_P
        self._initial_T = initial_T
        self._inlet_names: List[str] = []
        self._outlet_names: List[str] = []
        self._has_thermal_port = False
        super().__init__(name)

    # ── Port registration helpers ───────────────────────────────────────────

    def add_inlet(self, port_name: str = "inlet") -> FluidPort:
        """Add a named fluid inlet port. Returns the new port."""
        p = FluidPort(port_name, PortDirection.INLET, self)
        self._register_port(port_name, p)
        self._inlet_names.append(port_name)
        return p

    def add_outlet(self, port_name: str = "outlet") -> FluidPort:
        """Add a named fluid outlet port. Returns the new port."""
        p = FluidPort(port_name, PortDirection.OUTLET, self)
        self._register_port(port_name, p)
        self._outlet_names.append(port_name)
        return p

    def add_thermal_port(self, port_name: str = "heat") -> ThermalPort:
        """Add optional thermal port to receive heat from a MetalNode."""
        p = ThermalPort(port_name, PortDirection.INLET, self)
        self._register_port(port_name, p)
        self._has_thermal_port = True
        return p

    # ── BaseComponent hooks ─────────────────────────────────────────────────

    def _declare_ports(self) -> None:
        # Ports are added dynamically; nothing declared here.
        pass

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
        R_eff = fs.cp - fs.cv  # J/(kg·K), specific gas constant equivalent

        # Net mass flow
        mdot_in  = sum(inputs.get(f"{n}.mdot", 0.0) for n in self._inlet_names)
        mdot_out = sum(inputs.get(f"{n}.mdot", 0.0) for n in self._outlet_names)
        mdot_net = mdot_in - mdot_out

        # Enthalpy flux
        h_in_flux  = sum(
            inputs.get(f"{n}.mdot", 0.0) * inputs.get(f"{n}.h", states["h"])
            for n in self._inlet_names
        )
        h_out_flux = sum(
            inputs.get(f"{n}.mdot", 0.0) * states["h"]
            for n in self._outlet_names
        )

        # External heat input
        Q_dot = inputs.get("heat.Q_dot", 0.0)

        # Pressure ODE: derived from ideal-gas-like volume conservation
        # Uses gamma*R*T/V = gamma*P/(rho*V) = gamma*P/m
        dP_dt = (fs.gamma * R_eff * fs.T / V) * (mdot_net / fs.rho)

        # Enthalpy ODE: energy balance
        dh_dt = (1.0 / m) * (Q_dot + h_in_flux - h_out_flux - V * dP_dt)

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

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def thermo(self) -> ThermoBackend:
        return self._thermo
