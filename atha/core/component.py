# atha/core/component.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from atha.core.port import Port


class BaseComponent(ABC):
    """
    Abstract base class for all engine components.

    Subclass contract:
      1. Call super().__init__(name) AFTER setting any instance attributes
         that _declare_* methods read.
      2. Implement all abstract methods.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._ports: Dict[str, Port] = {}
        self._state_names: List[str] = []
        self._state_values: Dict[str, float] = {}
        self._alg_names: List[str] = []
        self._params: Dict[str, Any] = {}
        self.last_outputs: Dict[str, Any] = {}
        self._declare_ports()
        self._declare_states()
        self._declare_algebraic_vars()

    # ── Subclass hooks ──────────────────────────────────────────────────────

    @abstractmethod
    def _declare_ports(self) -> None:
        """Register all ports via _register_port()."""

    @abstractmethod
    def _declare_states(self) -> None:
        """Register differential state variables via _register_state()."""

    @abstractmethod
    def _declare_algebraic_vars(self) -> None:
        """Register algebraic residual names via _register_algebraic()."""

    # ── Core evaluation interface ───────────────────────────────────────────

    @abstractmethod
    def compute_outputs(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Purely algebraic evaluation. Given states and inputs, compute outputs.
        Must be side-effect free on self._state_values.
        """

    @abstractmethod
    def get_state_derivatives(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Returns {state_name: dX/dt} for this component's differential states.
        Components with no states return {}.
        """

    @abstractmethod
    def get_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Algebraic residuals: {name: value} where each value must equal 0.
        Components with no algebraic constraints return {}.
        """

    @abstractmethod
    def initialize(self, operating_point: Dict[str, float]) -> None:
        """Set initial state values from a coarse operating point estimate."""

    # ── Registration helpers ────────────────────────────────────────────────

    def _register_state(self, name: str, initial_value: float = 0.0) -> None:
        self._state_names.append(name)
        self._state_values[name] = initial_value

    def _register_algebraic(self, name: str) -> None:
        self._alg_names.append(name)

    def _register_port(self, name: str, port: Port) -> None:
        self._ports[name] = port

    # ── Public accessors ────────────────────────────────────────────────────

    @property
    def ports(self) -> Dict[str, Port]:
        return self._ports

    def port(self, name: str) -> Port:
        if name not in self._ports:
            raise KeyError(f"Component '{self.name}' has no port '{name}'.")
        return self._ports[name]

    @property
    def n_states(self) -> int:
        return len(self._state_names)

    @property
    def state_names(self) -> List[str]:
        return list(self._state_names)

    @property
    def n_algebraic(self) -> int:
        return len(self._alg_names)

    @property
    def algebraic_names(self) -> List[str]:
        return list(self._alg_names)

    def get_steady_state_residuals(
        self,
        t: float,
        states: Dict[str, float],
        inputs: Dict[str, float],
        outputs: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """
        Optional dimensionless O(1) residuals for steady-state Newton.

        Return a dict {state_name: residual} where each value should be ~O(1)
        at nominal conditions and zero at steady state.  Return None (default)
        to fall back to scaled get_state_derivatives.
        """
        return None

    def get_jacobian_sparsity(self) -> Optional[Dict[Tuple[str, str], int]]:
        """Optional: return sparsity pattern for Jacobian optimization."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
