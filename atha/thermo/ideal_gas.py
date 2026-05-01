from __future__ import annotations
import numpy as np
from atha.thermo.interface import FluidState, ThermoBackend


class IdealGasBackend(ThermoBackend):
    """
    Analytic ideal gas with constant gamma and R.
    Used for unit testing and quick sanity checks without Cantera/CoolProp.
    """

    def __init__(self, gamma: float, R: float,
                 mu: float = 1.8e-5, k: float = 0.026, MW: float = 0.029):
        self.gamma = gamma
        self.R = R          # J/(kg·K), specific gas constant
        self.mu = mu
        self.k = k
        self.MW = MW
        self.cp = gamma * R / (gamma - 1)
        self.cv = R / (gamma - 1)

    def state_from_PT(self, P: float, T: float) -> FluidState:
        rho = P / (self.R * T)
        h = self.cp * T
        s = self.cp * np.log(T) - self.R * np.log(P)
        return FluidState(P=P, T=T, h=h, rho=rho, s=s,
                          cp=self.cp, cv=self.cv, gamma=self.gamma,
                          mu=self.mu, k=self.k, MW=self.MW,
                          phase='gas', quality=None)

    def state_from_Ph(self, P: float, h: float) -> FluidState:
        T = h / self.cp
        return self.state_from_PT(P, T)

    def state_from_Ps(self, P: float, s: float) -> FluidState:
        # s = cp*ln(T) - R*ln(P)  =>  T = exp((s + R*ln(P)) / cp)
        T = float(np.exp((s + self.R * np.log(P)) / self.cp))
        return self.state_from_PT(P, T)

    def isentropic_expansion(self, inlet: FluidState, P_exit: float) -> FluidState:
        T_exit = inlet.T * (P_exit / inlet.P) ** ((self.gamma - 1) / self.gamma)
        return self.state_from_PT(P_exit, T_exit)
