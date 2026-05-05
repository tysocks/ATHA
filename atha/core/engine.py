# atha/core/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from atha.core.component import BaseComponent
from atha.core.evaluation import EvaluationResult
from atha.core.port import Port, PortConnectionError, PortDomain
from atha.core.registry import ResidualRegistry, VariableKind, VariableRegistry


def _component_inputs(comp_name: str, commands: Dict[str, float]) -> Dict[str, float]:
    """Return global commands plus component-prefixed keys stripped for local use."""
    prefix = comp_name + "."
    inputs: Dict[str, float] = {}
    for key, value in commands.items():
        inputs[key] = value
        if key.startswith(prefix):
            inputs[key[len(prefix):]] = value
    return inputs


@dataclass(frozen=True)
class Connection:
    """Directed port connection stored in EngineLayout for solver propagation."""
    src_comp: str
    src_port: str
    dst_comp: str
    dst_port: str
    domain: PortDomain


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
        connections: Optional[List[Connection]] = None,
        variable_registry: Optional[VariableRegistry] = None,
        residual_registry: Optional[ResidualRegistry] = None,
    ) -> None:
        self.components = components
        self.state_offsets = state_offsets
        self.alg_offsets = alg_offsets
        self.n_states = n_states
        self.n_algebraic = n_algebraic
        self.connections: List[Connection] = connections or []
        self.variable_registry = variable_registry or VariableRegistry()
        self.residual_registry = residual_registry or ResidualRegistry()

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

    def all_algebraic_names(self) -> List[str]:
        """Return ordered list of 'component.algebraic' labels."""
        names = []
        for comp in self.components:
            off = self.alg_offsets.get(comp.name)
            if off is not None:
                for aname in comp.algebraic_names:
                    names.append(f"{comp.name}.{aname}")
        return names

    def evaluate(self, t: float, X: np.ndarray, Z: np.ndarray, U: Dict[str, float]) -> EvaluationResult:
        """
        Evaluate component derivatives, algebraic residuals, and telemetry.

        This is the Phase 1 compatibility surface for the future DAE assembly
        model. It preserves the existing component compute contract while
        returning named residuals and outputs in a structured result.
        """
        dXdt = np.zeros(self.n_states)
        residual_values: Dict[str, float] = {}
        outputs_all: Dict[str, float] = {}
        outputs_by_comp: Dict[str, Dict[str, Any]] = {}
        states_by_comp: Dict[str, Dict[str, float]] = {}
        inputs_by_comp: Dict[str, Dict[str, float]] = {}

        for comp in self.components:
            off = self.state_offsets.get(comp.name)
            states = {}
            if off is not None:
                for i, name in enumerate(comp.state_names):
                    states[name] = float(X[off + i])
            states_by_comp[comp.name] = states

            alg_off = self.alg_offsets.get(comp.name)
            inputs = _component_inputs(comp.name, U)
            if alg_off is not None:
                for i, name in enumerate(comp.algebraic_names):
                    inputs[name] = float(Z[alg_off + i])
            inputs_by_comp[comp.name] = inputs

            outputs = comp.compute_outputs(t, states, inputs)
            comp.last_outputs = dict(outputs)
            outputs_by_comp[comp.name] = outputs
            for name, value in outputs.items():
                outputs_all[f"{comp.name}.{name}"] = value

            derivs = comp.get_state_derivatives(t, states, inputs, outputs)
            if off is not None:
                for i, name in enumerate(comp.state_names):
                    dXdt[off + i] = derivs.get(name, 0.0)

            residuals = comp.get_residuals(t, states, inputs, outputs)
            for name, value in residuals.items():
                residual_values[f"{comp.name}.{name}"] = value

        residual_values.update(
            self._evaluate_connection_residuals(outputs_by_comp, states_by_comp, inputs_by_comp, U)
        )

        residual_names = self.residual_registry.names()
        Rz = np.array([residual_values.get(name, 0.0) for name in residual_names], dtype=float)
        return EvaluationResult(
            dXdt=dXdt,
            Rz=Rz,
            outputs=outputs_all,
            residual_names=residual_names,
            output_names=list(outputs_all.keys()),
            residual_scales=np.array(self.residual_registry.scales(), dtype=float),
        )

    def _evaluate_connection_residuals(
        self,
        outputs_by_comp: Dict[str, Dict[str, Any]],
        states_by_comp: Dict[str, Dict[str, float]],
        inputs_by_comp: Dict[str, Dict[str, float]],
        commands: Dict[str, float],
    ) -> Dict[str, float]:
        residuals: Dict[str, float] = {}
        for conn in self.connections:
            source_name = conn.src_comp
            sink_name = conn.dst_comp
            base = f"connection.{source_name}.{conn.src_port}__{sink_name}.{conn.dst_port}"

            if conn.domain == PortDomain.FLUID:
                props = ("P", "h", "mdot")
                combine = lambda source_v, sink_v: sink_v - source_v
            elif conn.domain == PortDomain.SHAFT:
                props = ("omega", "tau")
                combine = lambda source_v, sink_v: source_v + sink_v if current_prop == "tau" else sink_v - source_v
            elif conn.domain == PortDomain.THERMAL:
                props = ("T_wall", "Q_dot")
                combine = lambda source_v, sink_v: source_v + sink_v if current_prop == "Q_dot" else sink_v - source_v
            else:
                props = ()

            for current_prop in props:
                source_v = self._lookup_connection_value(
                    source_name, conn.src_port, current_prop, outputs_by_comp, states_by_comp, inputs_by_comp, commands
                )
                sink_v = self._lookup_connection_value(
                    sink_name, conn.dst_port, current_prop, outputs_by_comp, states_by_comp, inputs_by_comp, commands
                )
                if source_v is None or sink_v is None:
                    residuals[f"{base}.{current_prop}"] = 0.0
                else:
                    residuals[f"{base}.{current_prop}"] = float(combine(float(source_v), float(sink_v)))
        return residuals

    @staticmethod
    def _lookup_connection_value(
        comp_name: str,
        port_name: str,
        prop: str,
        outputs_by_comp: Dict[str, Dict[str, Any]],
        states_by_comp: Dict[str, Dict[str, float]],
        inputs_by_comp: Dict[str, Dict[str, float]],
        commands: Dict[str, float],
    ) -> Optional[float]:
        outputs = outputs_by_comp.get(comp_name, {})
        states = states_by_comp.get(comp_name, {})
        inputs = inputs_by_comp.get(comp_name, {})
        keys = (
            f"{port_name}.{prop}",
            prop,
        )
        for key in keys:
            if key in outputs and isinstance(outputs[key], (int, float, np.floating)):
                return float(outputs[key])
        if prop in states:
            return float(states[prop])
        for key in keys:
            if key in inputs:
                return float(inputs[key])
        for key in (f"{comp_name}.{port_name}.{prop}", f"{comp_name}.{prop}"):
            if key in commands:
                return float(commands[key])
        return None


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
        variable_registry = VariableRegistry()
        residual_registry = ResidualRegistry()
        state_idx = 0
        alg_idx = 0

        for comp in components:
            if comp.n_states > 0:
                state_offsets[comp.name] = state_idx
                for name in comp.state_names:
                    variable_registry.register(
                        f"{comp.name}.{name}",
                        VariableKind.STATE,
                        units="",
                        scale=1.0,
                        owner=comp.name,
                    )
                state_idx += comp.n_states
            if comp.n_algebraic > 0:
                alg_offsets[comp.name] = alg_idx
                for name in comp.algebraic_names:
                    full_name = f"{comp.name}.{name}"
                    variable_registry.register(
                        full_name,
                        VariableKind.ALGEBRAIC,
                        units="",
                        scale=1.0,
                        owner=comp.name,
                    )
                    residual_registry.register(
                        full_name,
                        units="",
                        scale=1.0,
                        owner=comp.name,
                    )
                alg_idx += comp.n_algebraic

        connections: List[Connection] = []
        for src_port, dst_port in self._connections:
            connections.append(Connection(
                src_comp=src_port.owner.name,
                src_port=src_port.name,
                dst_comp=dst_port.owner.name,
                dst_port=dst_port.name,
                domain=src_port.domain,
            ))

        for conn in connections:
            base = f"connection.{conn.src_comp}.{conn.src_port}__{conn.dst_comp}.{conn.dst_port}"
            if conn.domain == PortDomain.FLUID:
                for name, units, scale in (("P", "Pa", 1e6), ("h", "J/kg", 1e6), ("mdot", "kg/s", 1.0)):
                    residual_registry.register(f"{base}.{name}", units=units, scale=scale, owner="connection")
            elif conn.domain == PortDomain.SHAFT:
                for name, units, scale in (("omega", "rad/s", 1000.0), ("tau", "N*m", 100.0)):
                    residual_registry.register(f"{base}.{name}", units=units, scale=scale, owner="connection")
            elif conn.domain == PortDomain.THERMAL:
                for name, units, scale in (("T_wall", "K", 1000.0), ("Q_dot", "W", 1000.0)):
                    residual_registry.register(f"{base}.{name}", units=units, scale=scale, owner="connection")

        self._compiled = True
        return EngineLayout(
            components=components,
            state_offsets=state_offsets,
            alg_offsets=alg_offsets,
            n_states=state_idx,
            n_algebraic=alg_idx,
            connections=connections,
            variable_registry=variable_registry,
            residual_registry=residual_registry,
        )

    def __getitem__(self, name: str) -> BaseComponent:
        return self._components[name]

    def __contains__(self, name: str) -> bool:
        return name in self._components
