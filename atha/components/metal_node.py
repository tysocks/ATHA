# atha/components/metal_node.py
from __future__ import annotations
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import ThermalPort, PortDirection


class MetalNode(BaseComponent):
    """
    Lumped thermal capacitance (metal wall node).

    Physics:
        dT_wall/dt = (Q_dot_hot - Q_dot_cool) / (mass * cp_metal)

    State: T_wall [K]

    Ports:
        hot_side  (ThermalPort, INLET)  — receives heat from hot gas
        cool_side (ThermalPort, OUTLET) — gives heat to coolant

    Parameters:
        mass      [kg]
        cp_metal  [J/(kg·K)]  default 500.0 (steel)
        initial_T [K]         default 300.0
    """

    def __init__(
        self,
        name: str,
        mass: float,
        cp_metal: float = 500.0,
        initial_T: float = 300.0,
    ) -> None:
        self._mass = mass
        self._cp_metal = cp_metal
        self._initial_T = initial_T
        super().__init__(name)

    def _declare_ports(self) -> None:
        self._register_port("hot_side", ThermalPort("hot_side", PortDirection.INLET, self))
        self._register_port("cool_side", ThermalPort("cool_side", PortDirection.OUTLET, self))

    def _declare_states(self) -> None:
        self._register_state("T_wall", self._initial_T)

    def _declare_algebraic_vars(self) -> None:
        pass

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {"T_wall": states["T_wall"]}

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        Q_hot = inputs.get("hot_side.Q_dot", 0.0)
        Q_cool = inputs.get("cool_side.Q_dot", 0.0)
        dT_dt = (Q_hot - Q_cool) / (self._mass * self._cp_metal)
        return {"T_wall": dT_dt}

    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        T = operating_point.get("T_wall", self._initial_T)
        self._state_values["T_wall"] = T
