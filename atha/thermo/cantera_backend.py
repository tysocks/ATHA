from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cantera as ct
from atha.thermo.interface import FluidState, ThermoBackend


@dataclass(frozen=True)
class CombustionResult:
    """Output of an adiabatic equilibrium combustion calculation."""
    T_ad:       float   # K, adiabatic flame temperature
    gamma:      float   # effective isentropic exponent of products
    MW:         float   # kg/mol, mean molecular weight of products
    h_products: float   # J/kg, specific enthalpy of products
    c_star:     float   # m/s, characteristic velocity
    cp:         float   # J/(kg·K), Cp of products


class CanteraBackend(ThermoBackend):
    """
    Combustion thermochemistry via Cantera.

    Primary use: adiabatic equilibrium combustion for chambers and preburners.
    Also implements ThermoBackend for hot-gas product state lookups.

    mechanism: Cantera YAML mechanism file, e.g. 'h2o2.yaml' for H2/O2,
               'gri30.yaml' for hydrocarbon combustion.
    """

    def __init__(self, mechanism: str = "h2o2.yaml"):
        self._gas = ct.Solution(mechanism)

    def combustion_equilibrium(
        self,
        fuel: str,
        oxidizer: str,
        MR: float,          # oxidizer-to-fuel mass ratio
        P_chamber: float,   # Pa
    ) -> CombustionResult:
        """
        Adiabatic constant-pressure equilibrium combustion.

        MR = mass_oxidizer / mass_fuel
        Uses Cantera HP equilibrium (constant enthalpy and pressure).
        """
        gas = self._gas
        Y_fuel = 1.0 / (1.0 + MR)
        Y_ox   = MR  / (1.0 + MR)
        gas.Y = {fuel: Y_fuel, oxidizer: Y_ox}
        gas.TP = 300.0, P_chamber
        gas.equilibrate('HP')

        g = gas.cp_mass / gas.cv_mass
        R_eff = ct.gas_constant / gas.mean_molecular_weight
        # c* from isentropic throat relation
        c_star = (np.sqrt(g * R_eff * gas.T) /
                  (g * np.sqrt((2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0)))))

        return CombustionResult(
            T_ad=float(gas.T),
            gamma=float(g),
            MW=float(gas.mean_molecular_weight),
            h_products=float(gas.enthalpy_mass),
            c_star=float(c_star),
            cp=float(gas.cp_mass),
        )

    # ── ThermoBackend implementation for hot-gas product states ──────────────

    def state_from_PT(self, P: float, T: float) -> FluidState:
        self._gas.TP = T, P
        return self._to_fluid_state()

    def state_from_Ph(self, P: float, h: float) -> FluidState:
        self._gas.HP = h, P
        return self._to_fluid_state()

    def state_from_Ps(self, P: float, s: float) -> FluidState:
        self._gas.SP = s, P
        return self._to_fluid_state()

    def isentropic_expansion(self, inlet: FluidState, P_exit: float) -> FluidState:
        return self.state_from_Ps(P_exit, inlet.s)

    def _to_fluid_state(self) -> FluidState:
        g = self._gas
        gamma = g.cp_mass / g.cv_mass
        return FluidState(
            P=float(g.P), T=float(g.T),
            h=float(g.enthalpy_mass), rho=float(g.density),
            s=float(g.entropy_mass), cp=float(g.cp_mass), cv=float(g.cv_mass),
            gamma=float(gamma), mu=float(g.viscosity),
            k=float(g.thermal_conductivity),
            MW=float(g.mean_molecular_weight),
            phase='gas', quality=None,
        )
