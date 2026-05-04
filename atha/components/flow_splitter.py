# atha/components/flow_splitter.py
from __future__ import annotations
from typing import Dict

from atha.core.component import BaseComponent
from atha.core.port import FluidPort, PortDirection


class FlowSplitter(BaseComponent):
    """
    Ideal mass-flow tee: identical fluid state on both branches, split mass flows.

    ``split_fraction`` is the fraction of inlet mass flow sent to ``outlet_a``;
    ``outlet_b`` receives ``1 - split_fraction``.
    """

    def __init__(self, name: str, split_fraction: float = 0.5) -> None:
        self._split = float(split_fraction)
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("inlet", FluidPort("inlet", PortDirection.INLET, self))
        self._register_port("outlet_a", FluidPort("outlet_a", PortDirection.OUTLET, self))
        self._register_port("outlet_b", FluidPort("outlet_b", PortDirection.OUTLET, self))

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
        P = inputs.get("inlet.P", 1e5)
        h = inputs.get("inlet.h", 0.0)
        mdot_in = inputs.get("inlet.mdot", 0.0)
        f = self._split
        return {
            "outlet_a.P": P,
            "outlet_a.h": h,
            "outlet_a.mdot": f * mdot_in,
            "outlet_b.P": P,
            "outlet_b.h": h,
            "outlet_b.mdot": (1.0 - f) * mdot_in,
        }

    def get_state_derivatives(self, t, states, inputs, outputs):
        return {}

    def get_residuals(self, t, states, inputs, outputs):
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        pass
