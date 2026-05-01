# atha/core/port.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from atha.core.component import BaseComponent


class PortDomain(Enum):
    FLUID   = auto()
    SHAFT   = auto()
    THERMAL = auto()


class PortDirection(Enum):
    INLET  = auto()
    OUTLET = auto()


class PortConnectionError(ValueError):
    pass


@dataclass
class Port:
    name: str
    direction: PortDirection
    owner: Optional["BaseComponent"]
    _connected_to: Optional["Port"] = field(default=None, repr=False, compare=False)

    @property
    def domain(self) -> PortDomain:
        raise NotImplementedError("Subclasses must define domain.")

    def connect(self, other: "Port") -> None:
        if self.domain != other.domain:
            raise PortConnectionError(
                f"Cannot connect {self.name} ({self.domain.name}) "
                f"to {other.name} ({other.domain.name}): domain mismatch."
            )
        if self.direction == other.direction:
            raise PortConnectionError(
                f"Cannot connect two {self.direction.name} ports: "
                f"{self.name} and {other.name}."
            )
        if self._connected_to is not None:
            raise PortConnectionError(
                f"Port {self.name} is already connected to {self._connected_to.name}."
            )
        if other._connected_to is not None:
            raise PortConnectionError(
                f"Port {other.name} is already connected to {other._connected_to.name}."
            )
        self._connected_to = other
        other._connected_to = self

    @property
    def is_connected(self) -> bool:
        return self._connected_to is not None

    @property
    def connected_to(self) -> Optional["Port"]:
        return self._connected_to


@dataclass
class FluidPort(Port):
    # Live values updated by solver during evaluation
    mdot: float = field(default=0.0, compare=False)  # kg/s
    P:    float = field(default=0.0, compare=False)  # Pa
    h:    float = field(default=0.0, compare=False)  # J/kg

    @property
    def domain(self) -> PortDomain:
        return PortDomain.FLUID


@dataclass
class ShaftPort(Port):
    omega: float = field(default=0.0, compare=False)  # rad/s
    tau:   float = field(default=0.0, compare=False)  # N·m

    @property
    def domain(self) -> PortDomain:
        return PortDomain.SHAFT


@dataclass
class ThermalPort(Port):
    T_wall: float = field(default=300.0, compare=False)  # K
    Q_dot:  float = field(default=0.0,   compare=False)  # W

    @property
    def domain(self) -> PortDomain:
        return PortDomain.THERMAL
