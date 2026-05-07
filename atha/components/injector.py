from __future__ import annotations

from typing import Dict, Optional

from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection


class MassFlowInjector(BaseComponent):
    """
    Minimal injector model for runbox-style studies.

    Purpose
    -------
    - Accept a commanded mass flow rate (as ``inlet.mdot``).
    - Propagate fluid properties downstream to the chamber.
    - Optionally compute an implied pressure drop if ``outlet.P`` is available.

    This is intentionally simple: it does not attempt to compute mdot from CdA.

    Ports
    -----
    inlet   FluidPort INLET
    outlet  FluidPort OUTLET
    """

    def __init__(self, name: str, delta_P_nominal: float = 0.0) -> None:
        self._delta_P_nominal = float(delta_P_nominal)
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
        _ = (t, states)
        P_in = float(inputs.get("inlet.P", 0.0))
        h_in = float(inputs.get("inlet.h", 0.0))
        T_in = float(inputs.get("inlet.T", 0.0))
        rho_in = float(inputs.get("inlet.rho", 0.0))
        gamma_in = float(inputs.get("inlet.gamma", 0.0))
        mdot_cmd = float(inputs.get("inlet.mdot", 0.0))

        # If an outlet pressure is present (e.g. seeded from chamber.P), use it.
        # Otherwise, apply a nominal pressure drop from inlet.
        P_out: Optional[float] = inputs.get("outlet.P", None)
        if P_out is None:
            P_out = max(P_in - self._delta_P_nominal, 1.0)
        else:
            P_out = float(P_out)

        delta_P = max(P_in - P_out, 0.0)

        return {
            # Port-qualified outputs
            "outlet.P": P_out,
            "outlet.h": h_in,
            "outlet.T": T_in,
            "outlet.rho": rho_in,
            "outlet.gamma": gamma_in,
            "outlet.mdot": mdot_cmd,
            # Bare keys for connection propagation convenience
            "P": P_out,
            "h": h_in,
            "T": T_in,
            "rho": rho_in,
            "gamma": gamma_in,
            "mdot": mdot_cmd,
            "delta_P": delta_P,
        }

    def get_state_derivatives(self, t, states, inputs, outputs):
        return {}

    def get_residuals(self, t, states, inputs, outputs):
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        pass

