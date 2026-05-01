# atha/jannaf/simplified.py
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.jannaf.performance import PerformanceResult
from atha.thermo.interface import ThermoBackend

G0 = 9.80665  # m/s^2, standard gravity


class SimplifiedJANNAF:
    """
    JANNAF Simplified Analytical Procedure for rocket nozzle performance.
    Reference: JANNAF CPIA-246, Section 3.

    Calculation chain:
        1. Ideal c* from ThermoBackend (or given T_chamber and gamma)
        2. Apply eta_cstar -> delivered c*
        3. Compute ideal Cf from isentropic nozzle expansion
        4. Apply Cf efficiency factors -> delivered Cf
        5. Isp = c* * Cf / g0
        6. Thrust = Cf * Pc * At
    """

    def __init__(
        self,
        thermo: ThermoBackend,
        efficiencies: JANNAFEfficiencies,
        throat_area: float,           # m^2
        exit_area: float,             # m^2
        ambient_pressure: float = 0.0,  # Pa, 0 = vacuum
    ) -> None:
        self._thermo = thermo
        self.efficiencies = efficiencies
        self.A_t = throat_area
        self.A_e = exit_area
        self.P_a = ambient_pressure
        self.epsilon = exit_area / throat_area  # expansion ratio

    def compute(
        self,
        P_chamber: float,    # Pa
        T_chamber: float,    # K, adiabatic flame temperature
        MR: float,           # oxidizer/fuel mass ratio (for reporting only)
        mdot_total: float,   # kg/s (for reporting only)
    ) -> PerformanceResult:
        """
        Compute delivered nozzle performance using the JANNAF simplified procedure.

        P_chamber and T_chamber must be the combustion chamber conditions
        (not stagnation — for well-stirred reactors these are approximately equal).
        """
        fs = self._thermo.state_from_PT(P_chamber, T_chamber)
        gamma = fs.gamma
        R_eff = fs.cp - fs.cv   # J/(kg·K)

        # Step 1: Ideal c* from throat choked flow relation
        # c* = sqrt(gamma*R*T) / (gamma * sqrt((2/(gamma+1))^((gamma+1)/(gamma-1))))
        term = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        c_star_ideal = np.sqrt(gamma * R_eff * T_chamber) / (gamma * np.sqrt(term))

        # Step 2: Delivered c*
        c_star_del = self.efficiencies.eta_cstar * c_star_ideal

        # Step 3: Exit Mach number from area ratio (supersonic branch)
        Me = self._exit_mach(gamma, self.epsilon)

        # Step 4: Exit conditions
        Pe = P_chamber * (1.0 + (gamma - 1.0) / 2.0 * Me**2) ** (-gamma / (gamma - 1.0))
        Te = T_chamber / (1.0 + (gamma - 1.0) / 2.0 * Me**2)
        Ve = Me * np.sqrt(gamma * R_eff * Te)

        # Step 5: Ideal Cf (JANNAF formula)
        Cf_ideal = self._ideal_cf(gamma, P_chamber, Pe, self.P_a, self.epsilon)

        # Step 6: Delivered Cf
        Cf_del = self.efficiencies.overall_Cf_efficiency * Cf_ideal

        # Step 7: Performance metrics
        Isp_ideal = c_star_ideal * Cf_ideal / G0
        Isp_del   = c_star_del   * Cf_del   / G0
        thrust    = Cf_del * P_chamber * self.A_t

        return PerformanceResult(
            Isp=Isp_del,
            c_star=c_star_del,
            Cf=Cf_del,
            thrust=thrust,
            mdot_total=mdot_total,
            c_star_ideal=c_star_ideal,
            Isp_ideal=Isp_ideal,
            MR=MR,
            P_chamber=P_chamber,
            T_chamber=T_chamber,
            expansion_ratio=self.epsilon,
            P_exit=Pe,
            V_exit=Ve,
            efficiencies=self.efficiencies,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _ideal_cf(
        gamma: float,
        Pc: float,
        Pe: float,
        Pa: float,
        epsilon: float,
    ) -> float:
        """
        JANNAF ideal thrust coefficient formula.
        Cf = sqrt(2*gamma^2/(gamma-1) * (2/(gamma+1))^((gamma+1)/(gamma-1))
                  * [1 - (Pe/Pc)^((gamma-1)/gamma)])
             + (Pe - Pa)/Pc * epsilon
        """
        g = gamma
        pr = Pe / Pc
        momentum_term = np.sqrt(
            2.0 * g**2 / (g - 1.0)
            * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
            * (1.0 - pr ** ((g - 1.0) / g))
        )
        pressure_term = (Pe - Pa) / Pc * epsilon
        return momentum_term + pressure_term

    @staticmethod
    def _exit_mach(gamma: float, epsilon: float) -> float:
        """Solve area-Mach relation for supersonic exit Mach number."""
        def area_mach(M: float) -> float:
            return (
                (1.0 / M)
                * ((2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M**2))
                ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
            ) - epsilon
        if epsilon <= 1.0 + 1e-6:
            return 1.0
        return brentq(area_mach, 1.001, 100.0, xtol=1e-10)
