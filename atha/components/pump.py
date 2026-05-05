# atha/components/pump.py
from __future__ import annotations
import math
from typing import Dict, Optional
from atha.core.component import BaseComponent
from atha.core.port import FluidPort, ShaftPort, PortDirection


class PumpMap:
    """
    Design-point based centrifugal pump performance map.

    Uses affinity laws:
        ΔP ∝ ω²
        ṁ  ∝ ω  (at constant flow coefficient)
        η  = η_design (constant; extend to a curve if needed)
    """

    def __init__(
        self,
        mdot_design: float,
        dP_design: float,
        omega_design: float,   # rad/s
        eta_design: float,
    ) -> None:
        self._mdot_design = mdot_design
        self._dP_design = dP_design
        self._omega_design = omega_design
        self._eta_design = eta_design

    @classmethod
    def from_design_point(
        cls,
        mdot_design: float,
        dP_design: float,
        speed_design: float,   # rpm
        efficiency_design: float,
        fluid_density: float = None,  # optional; reserved for map calibration
        **kwargs,
    ) -> "PumpMap":
        _ = (fluid_density, kwargs)  # API compatibility with example scripts
        return cls(mdot_design, dP_design, speed_design * math.pi / 30.0, efficiency_design)

    def delta_P(self, omega: float) -> float:
        """Pressure rise at shaft speed omega [rad/s]."""
        ratio = omega / max(self._omega_design, 1.0)
        return self._dP_design * ratio ** 2

    def efficiency(self, mdot: float = 0.0, omega: float = 0.0) -> float:
        return self._eta_design

    def scale_efficiency(self, factor: float) -> "PumpMap":
        return PumpMap(self._mdot_design, self._dP_design, self._omega_design,
                       min(self._eta_design * factor, 1.0))

    @property
    def omega_design(self) -> float:
        return self._omega_design

    @property
    def dP_design(self) -> float:
        return self._dP_design


class Pump(BaseComponent):
    """
    Centrifugal pump with PumpMap performance model.

    Physics:
        delta_P = pump_map.delta_P(omega)
        W       = mdot * delta_P / (rho * eta)
        tau     = W / max(omega, 1.0)   [shaft load]

    The pump reports pressure rise based on current shaft speed.
    Density is obtained from the coolant fluid at inlet conditions.

    Ports
    -----
    inlet   FluidPort INLET
    outlet  FluidPort OUTLET
    shaft   ShaftPort INLET   — receives shaft speed ω

    Additional outlet ports are created on demand via port("bleed"),
    port("bleed_ox"), port("bleed_fuel"), etc.  Each bleed port has the
    same outlet pressure as the main outlet.
    """

    def __init__(
        self,
        name: str,
        diameter: float,
        pump_map: PumpMap,
        fluid,
        efficiency_map=None,
    ) -> None:
        self._diameter = diameter
        self._pump_map = pump_map
        self._fluid = fluid
        self._efficiency_map = efficiency_map
        self.map_efficiency_scale: float = 1.0
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("inlet",  FluidPort("inlet",  PortDirection.INLET,  self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))
        self._register_port("shaft",  ShaftPort("shaft",  PortDirection.INLET,  self))

    def _declare_states(self) -> None:
        pass

    def _declare_algebraic_vars(self) -> None:
        pass

    def port(self, name: str) -> "Port":
        if name not in self._ports:
            # Auto-create any bleed outlet port on demand
            self._register_port(name, FluidPort(name, PortDirection.OUTLET, self))
        return self._ports[name]

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        omega = inputs.get("shaft.omega", self._pump_map.omega_design)
        P_in  = inputs.get("inlet.P", 4e5)
        h_in  = inputs.get("inlet.h", 0.0)
        mdot  = inputs.get("inlet.mdot", 0.0)

        # Fluid state at inlet (density, temperature for downstream propagation)
        try:
            _fs = self._fluid.state_from_Ph(P_in, h_in)
            rho = _fs.rho
            T_in = _fs.T
        except Exception:
            rho = 1000.0
            T_in = 300.0

        delta_P = self._pump_map.delta_P(omega)
        P_out = P_in + delta_P

        if self._efficiency_map is not None:
            ctx = dict(states)
            ctx.update(inputs)
            ctx["delta_P"] = delta_P
            eta = self._efficiency_map.evaluate(ctx)["efficiency"]
        else:
            eta = self._pump_map.efficiency(mdot, omega)

        eta = min(max(eta * self.map_efficiency_scale, 1e-6), 1.0)

        W   = abs(mdot) * delta_P / (rho * max(eta, 1e-6))
        tau = W / max(abs(omega), 1.0)

        return {
            "outlet.P":   P_out,
            "outlet.h":   h_in,   # no enthalpy rise in ideal pump
            "outlet.T":   T_in,
            "outlet.rho": rho,
            # Bare keys propagate through any port connection (e.g. bleed ports)
            "P":    P_out,
            "h":    h_in,
            "T":    T_in,
            "rho":  rho,
            "delta_P":    delta_P,
            "power":      W,
            "tau_load":   tau,
            "efficiency": eta,
        }

    def get_state_derivatives(self, t, states, inputs, outputs):
        return {}

    def get_residuals(self, t, states, inputs, outputs):
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        pass
