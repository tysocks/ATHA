from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FluidState:
    """Immutable thermodynamic state snapshot. All quantities SI."""
    P:       float          # Pa, pressure
    T:       float          # K, temperature
    h:       float          # J/kg, specific enthalpy
    rho:     float          # kg/m^3, density
    s:       float          # J/(kg·K), specific entropy
    cp:      float          # J/(kg·K), specific heat at constant pressure
    cv:      float          # J/(kg·K), specific heat at constant volume
    gamma:   float          # Cp/Cv, isentropic exponent
    mu:      float          # Pa·s, dynamic viscosity
    k:       float          # W/(m·K), thermal conductivity
    MW:      float          # kg/mol, molecular weight
    phase:   str            # 'liquid'|'gas'|'two-phase'|'supercritical'
    quality: Optional[float]  # vapor quality [0,1] for two-phase, else None


class ThermoBackend(ABC):
    """Abstract interface for all thermophysical property calculations."""

    @abstractmethod
    def state_from_Ph(self, P: float, h: float) -> FluidState:
        """Primary hot-path call during ODE integration (P and h are states)."""
        ...

    @abstractmethod
    def state_from_PT(self, P: float, T: float) -> FluidState:
        """Used for boundary condition specification."""
        ...

    @abstractmethod
    def state_from_Ps(self, P: float, s: float) -> FluidState:
        """Used for isentropic process calculations."""
        ...

    @abstractmethod
    def isentropic_expansion(self, inlet: FluidState, P_exit: float) -> FluidState:
        """Compute ideal isentropic exit state at P_exit."""
        ...
