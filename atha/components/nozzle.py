# atha/components/nozzle.py
from __future__ import annotations
from typing import Dict, Optional
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies

G0 = 9.80665  # m/s²


class Nozzle(BaseComponent):
    """
    Isentropic nozzle with choked throat and JANNAF efficiency factors.

    Ports
    -----
    inlet   FluidPort INLET   — chamber gas
    outlet  FluidPort OUTLET  — exhaust

    Parameters
    ----------
    throat_area      m²
    exit_area        m²
    efficiencies     JANNAFEfficiencies  (optional; uses defaults if omitted)
    gamma            -     default hot-gas gamma for Cf calculation (default 1.2)
    ambient_pressure Pa    default ambient; overridden by BCS key "P_ambient"

    Optional PerformanceMap overrides
    ----------------------------------
    cf_map               PerformanceMap  — replaces full Cf calculation; must output "Cf"
    discharge_coeff_map  PerformanceMap  — replaces scalar Cd within JANNAF calc
    """

    def __init__(
        self,
        name: str,
        throat_area: float,
        exit_area: float,
        efficiencies: Optional[JANNAFEfficiencies] = None,
        gamma: float = 1.2,
        ambient_pressure: float = 0.0,
        # Legacy scalar params (ignored when efficiencies is provided)
        discharge_coeff: float = 0.98,
        eta_velocity: float = 0.99,
        eta_divergence: float = 0.983,
        cf_map=None,
        discharge_coeff_map=None,
        # Legacy thermo (not used in map-free path)
        thermo=None,
    ) -> None:
        self._throat_area = throat_area
        self._exit_area = exit_area
        self._epsilon = exit_area / throat_area
        self._gamma_default = gamma
        self._ambient_pressure = ambient_pressure
        self._cf_map = cf_map
        self._discharge_coeff_map = discharge_coeff_map
        self._thermo = thermo

        if efficiencies is not None:
            self._eff = efficiencies
        else:
            self._eff = JANNAFEfficiencies(
                eta_Cd=discharge_coeff,
                eta_velocity=eta_velocity,
                eta_divergence=eta_divergence,
            )
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("inlet",  FluidPort("inlet",  PortDirection.INLET,  self))
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
        P_c  = inputs.get("inlet.P", 1e6)
        h_c  = inputs.get("inlet.h", 1e6)
        mdot = inputs.get("inlet.mdot", 0.0)

        # P_ambient from BCS key (supports both namespaced and bare forms)
        P_a = inputs.get("P_ambient",
               inputs.get("nozzle.P_ambient", self._ambient_pressure))

        context: Dict[str, float] = dict(states)
        context.update(inputs)
        context["ambient_pressure"] = P_a

        # Needed for choked mdot/c* even when Cf comes from a map.
        if self._thermo is not None:
            try:
                fs = self._thermo.state_from_Ph(P_c, h_c)
                gamma = fs.gamma
            except Exception:
                gamma = self._gamma_default
        else:
            gamma = inputs.get("inlet.gamma", self._gamma_default)
        if not isinstance(gamma, (int, float)):
            gamma = self._gamma_default
        gamma = float(gamma)
        gamma = min(max(gamma, 1.01), 2.0)

        if self._cf_map is not None:
            Cf_del = self._cf_map.evaluate(context)["Cf"]
            Cf_del_vac = Cf_del
        else:
            Me = SimplifiedJANNAF._exit_mach(gamma, self._epsilon)
            Pe = P_c * (1.0 + (gamma - 1.0) / 2.0 * Me ** 2) ** (-gamma / (gamma - 1.0))
            Cf_ideal = SimplifiedJANNAF._ideal_cf(gamma, P_c, Pe, P_a, self._epsilon)
            Cf_vac   = SimplifiedJANNAF._ideal_cf(gamma, P_c, Pe, 0.0, self._epsilon)

            if self._discharge_coeff_map is not None:
                Cd = self._discharge_coeff_map.evaluate(context)["discharge_coeff"]
            else:
                Cd = self._eff.eta_Cd

            Cf_del     = self._eff.eta_velocity * self._eff.eta_divergence * Cd * Cf_ideal
            Cf_del_vac = self._eff.eta_velocity * self._eff.eta_divergence * Cd * Cf_vac

        thrust     = Cf_del     * P_c * self._throat_area
        thrust_vac = Cf_del_vac * P_c * self._throat_area

        # Choked-throat mass flow from inlet density (ideal gas: rho = P/(R*T)).
        # Using rho directly avoids needing R separately:
        #   mdot_choked = Cd * At * P * sqrt(gamma/(R*T)) * Γ
        #               = Cd * At * sqrt(gamma * rho * P) * Γ   (since rho = P/(R*T))
        # Γ = (2/(γ+1))^((γ+1)/(2*(γ-1)))
        rho_c = inputs.get("inlet.rho", 0.0)
        if rho_c > 0.0:
            choked_exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
            choked_coeff = (2.0 / (gamma + 1.0)) ** choked_exp
            mdot_choked = (self._eff.eta_Cd * self._throat_area
                           * choked_coeff * (gamma * rho_c * P_c) ** 0.5)
        else:
            # Fall back to BCS-supplied mdot if rho unavailable
            mdot_choked = abs(mdot)

        mdot_eff = max(mdot_choked, 1e-10)
        c_star = P_c * self._throat_area / mdot_eff
        Isp_vacuum = thrust_vac / (mdot_eff * G0)

        return {
            "thrust":     thrust,
            "thrust_vac": thrust_vac,
            "Cf":         Cf_del,
            "c_star":     c_star,
            "Isp_vacuum": Isp_vacuum,
            "mdot":       mdot_choked,
        }

    def get_state_derivatives(self, t, states, inputs, outputs):
        return {}

    def get_residuals(self, t, states, inputs, outputs):
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        pass
