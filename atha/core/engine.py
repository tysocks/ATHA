# atha/core/engine.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from atha.core.component import BaseComponent
from atha.core.port import Port, PortConnectionError


class EngineLayout:
    """
    Immutable numerical description produced by Engine.compile().
    Solvers operate on this object exclusively — never on Engine directly.
    """

    def __init__(
        self,
        components: List[BaseComponent],       # in evaluation order
        state_offsets: Dict[str, int],         # component.name -> start index in X
        alg_offsets: Dict[str, int],           # component.name -> start index in Z
        n_states: int,
        n_algebraic: int,
    ) -> None:
        self.components = components
        self.state_offsets = state_offsets
        self.alg_offsets = alg_offsets
        self.n_states = n_states
        self.n_algebraic = n_algebraic

    def assemble_state_vector(self) -> np.ndarray:
        """Read current state values from all components into a flat array."""
        X = np.zeros(self.n_states)
        for comp in self.components:
            off = self.state_offsets.get(comp.name)
            if off is not None:
                for i, sname in enumerate(comp.state_names):
                    X[off + i] = comp._state_values[sname]
        return X

    def scatter_state_vector(self, X: np.ndarray) -> None:
        """Write flat state array back to component state dicts."""
        for comp in self.components:
            off = self.state_offsets.get(comp.name)
            if off is not None:
                for i, sname in enumerate(comp.state_names):
                    comp._state_values[sname] = float(X[off + i])

    def state_name_at(self, index: int) -> str:
        """Return 'component.state' label for global state index."""
        for comp in self.components:
            off = self.state_offsets.get(comp.name)
            if off is not None:
                for i, sname in enumerate(comp.state_names):
                    if off + i == index:
                        return f"{comp.name}.{sname}"
        raise IndexError(f"No state at index {index}")

    def all_state_names(self) -> List[str]:
        """Return ordered list of 'component.state' labels."""
        names = []
        for comp in self.components:
            off = self.state_offsets.get(comp.name)
            if off is not None:
                for sname in comp.state_names:
                    names.append(f"{comp.name}.{sname}")
        return names


class Engine:
    """
    Container for an engine topology. Build by adding components and
    connecting ports, then call compile() to get an EngineLayout for solvers.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._components: Dict[str, BaseComponent] = {}
        self._connections: List[Tuple[Port, Port]] = []
        self._compiled = False

    def add_component(self, component: BaseComponent) -> "Engine":
        """Register a component. Returns self for fluent chaining."""
        if component.name in self._components:
            raise ValueError(
                f"Component '{component.name}' already added to engine '{self.name}'."
            )
        self._components[component.name] = component
        return self

    def connect(self, source: Port, sink: Port) -> "Engine":
        """
        Connect two ports. Validates domain and direction compatibility.
        Returns self for fluent chaining.
        """
        source.connect(sink)   # raises PortConnectionError on mismatch
        self._connections.append((source, sink))
        return self

    def compile(self) -> EngineLayout:
        """
        Convert the component graph into an EngineLayout for numerical solvers.

        Assigns global indices to all state and algebraic variables.
        Uses component insertion order for evaluation (topological ordering
        is left to future work; for now insertion order is used which works
        for serial Newton/Radau solvers).
        """
        components = list(self._components.values())
        state_offsets: Dict[str, int] = {}
        alg_offsets: Dict[str, int] = {}
        state_idx = 0
        alg_idx = 0

        for comp in components:
            if comp.n_states > 0:
                state_offsets[comp.name] = state_idx
                state_idx += comp.n_states
            if comp.n_algebraic > 0:
                alg_offsets[comp.name] = alg_idx
                alg_idx += comp.n_algebraic

        self._compiled = True
        return EngineLayout(
            components=components,
            state_offsets=state_offsets,
            alg_offsets=alg_offsets,
            n_states=state_idx,
            n_algebraic=alg_idx,
        )

    def __getitem__(self, name: str) -> BaseComponent:
        return self._components[name]

    def __contains__(self, name: str) -> bool:
        return name in self._components
