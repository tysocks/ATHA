# atha/components/rotor.py
from __future__ import annotations
import math
from typing import Dict
from atha.core.component import BaseComponent
from atha.core.port import ShaftPort, PortDirection


class Rotor(BaseComponent):
    """
    Rotating shaft with angular momentum conservation.

    State: omega [rad/s]

    Physics:
        dω/dt = (Στ_drive - Στ_load - k_f * ω) / I

    Ports are created dynamically on first access.  Convention:
        turbine_in  (and any name containing "turbine") → INLET  (receives drive torque)
        pump_*      (and any other name)                → OUTLET (delivers omega to loads)

    Typical usage::

        shaft = Rotor("shaft", moment_of_inertia=0.10, initial_speed_rpm=28000)
        engine.connect(turbine.port("shaft"),   shaft.port("turbine_in"))
        engine.connect(lox_pump.port("shaft"),  shaft.port("pump_lox"))
        engine.connect(fuel_pump.port("shaft"), shaft.port("pump_fuel"))
    """

    def __init__(
        self,
        name: str,
        moment_of_inertia: float = None,
        initial_speed_rpm: float = 0.0,
        friction_coeff: float = 0.0,
        # Legacy aliases
        inertia: float = None,
        initial_omega: float = None,
    ) -> None:
        I = inertia if inertia is not None else moment_of_inertia
        if I is None:
            raise ValueError("Rotor requires moment_of_inertia (or legacy inertia kwarg)")
        omega0 = initial_omega if initial_omega is not None else initial_speed_rpm * math.pi / 30.0
        self._inertia = I
        self._friction_coeff = friction_coeff
        self._initial_omega = omega0
        self._turbine_port_names: list = []
        self._pump_port_names:    list = []
        super().__init__(name)
        self._state_values["omega"] = omega0

    def _declare_ports(self) -> None:
        pass  # all ports are created dynamically

    def _declare_states(self) -> None:
        self._register_state("omega", self._initial_omega)

    def _declare_algebraic_vars(self) -> None:
        pass

    def port(self, name: str) -> "Port":
        if name not in self._ports:
            if "turbine" in name.lower():
                direction = PortDirection.INLET
                self._turbine_port_names.append(name)
            else:
                direction = PortDirection.OUTLET
                self._pump_port_names.append(name)
            self._register_port(name, ShaftPort(name, direction, self))
        return self._ports[name]

    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        return {"omega": states.get("omega", self._initial_omega)}

    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        omega = states.get("omega", self._initial_omega)
        if "omega_override" in inputs:
            omega_t = float(inputs["omega_override"])
            k = 1e6
            return {"omega": k * (omega_t - omega)}
        tau_drive = sum(inputs.get(f"{n}.tau", 0.0) for n in self._turbine_port_names)
        tau_load  = sum(inputs.get(f"{n}.tau", 0.0) for n in self._pump_port_names)
        domega = (tau_drive - tau_load - self._friction_coeff * omega) / self._inertia
        return {"omega": domega}

    def get_residuals(self, t, states, inputs, outputs):
        return {}

    def initialize(self, operating_point: Dict[str, float]) -> None:
        if "omega" in operating_point:
            self._state_values["omega"] = operating_point["omega"]
        elif "rpm" in operating_point:
            self._state_values["omega"] = operating_point["rpm"] * math.pi / 30.0
