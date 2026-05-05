# tests/unit/test_core.py
"""Unit tests for core port, component, and engine infrastructure."""
import pytest
import numpy as np
from atha.core.port import (
    FluidPort, ShaftPort, ThermalPort,
    PortDirection, PortDomain, PortConnectionError,
)
from atha.core.component import BaseComponent
from atha.core.engine import Engine, EngineLayout
from atha.core.evaluation import EvaluationResult
from atha.core.registry import ResidualRegistry, VariableKind, VariableRegistry


# ── Minimal concrete component for testing ────────────────────────────────────

class _SimpleVolume(BaseComponent):
    """Two-state (P, h) test component with no algebraic constraints."""
    def __init__(self, name: str, initial_P: float = 1e5, initial_h: float = 3e5):
        self._initial_P = initial_P
        self._initial_h = initial_h
        super().__init__(name)

    def _declare_ports(self):
        self._register_port("inlet",  FluidPort("inlet",  PortDirection.INLET,  self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))

    def _declare_states(self):
        self._register_state("P", self._initial_P)
        self._register_state("h", self._initial_h)

    def _declare_algebraic_vars(self): pass

    def compute_outputs(self, t, states, inputs): return {"pressure": states["P"]}
    def get_state_derivatives(self, t, states, inputs, outputs): return {"P": 0.0, "h": 0.0}
    def get_residuals(self, t, states, inputs, outputs): return {}
    def initialize(self, op): pass


class _AlgebraicPipe(BaseComponent):
    """No-state algebraic component for testing."""
    def __init__(self, name: str):
        super().__init__(name)

    def _declare_ports(self):
        self._register_port("inlet",  FluidPort("inlet",  PortDirection.INLET,  self))
        self._register_port("outlet", FluidPort("outlet", PortDirection.OUTLET, self))

    def _declare_states(self): pass
    def _declare_algebraic_vars(self):
        self._register_algebraic("flow_balance")

    def compute_outputs(self, t, states, inputs): return {"mdot": inputs.get("mdot", 0.0)}
    def get_state_derivatives(self, t, states, inputs, outputs): return {}
    def get_residuals(self, t, states, inputs, outputs): return {"flow_balance": inputs.get("flow_residual", 0.0)}
    def initialize(self, op): pass


# ── Port tests ────────────────────────────────────────────────────────────────

def test_fluid_port_domain():
    p = FluidPort("test", PortDirection.INLET, None)
    assert p.domain == PortDomain.FLUID

def test_shaft_port_domain():
    p = ShaftPort("test", PortDirection.OUTLET, None)
    assert p.domain == PortDomain.SHAFT

def test_thermal_port_domain():
    p = ThermalPort("test", PortDirection.INLET, None)
    assert p.domain == PortDomain.THERMAL

def test_port_connect_valid():
    p_out = FluidPort("out", PortDirection.OUTLET, None)
    p_in  = FluidPort("in",  PortDirection.INLET,  None)
    p_out.connect(p_in)
    assert p_out.is_connected
    assert p_in.is_connected
    assert p_out.connected_to is p_in
    assert p_in.connected_to is p_out

def test_port_connect_domain_mismatch():
    fp = FluidPort("fp", PortDirection.OUTLET, None)
    sp = ShaftPort("sp", PortDirection.INLET, None)
    with pytest.raises(PortConnectionError, match="domain mismatch"):
        fp.connect(sp)

def test_port_connect_same_direction():
    p1 = FluidPort("p1", PortDirection.OUTLET, None)
    p2 = FluidPort("p2", PortDirection.OUTLET, None)
    with pytest.raises(PortConnectionError, match="OUTLET"):
        p1.connect(p2)

def test_port_connect_already_connected():
    p_out = FluidPort("out", PortDirection.OUTLET, None)
    p_in1 = FluidPort("in1", PortDirection.INLET,  None)
    p_in2 = FluidPort("in2", PortDirection.INLET,  None)
    p_out.connect(p_in1)
    with pytest.raises(PortConnectionError, match="already connected"):
        p_out.connect(p_in2)

def test_port_not_connected_by_default():
    p = FluidPort("p", PortDirection.INLET, None)
    assert not p.is_connected
    assert p.connected_to is None


# ── BaseComponent tests ───────────────────────────────────────────────────────

def test_base_component_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseComponent("test")

def test_component_name():
    v = _SimpleVolume("vol1")
    assert v.name == "vol1"

def test_component_state_count():
    v = _SimpleVolume("vol")
    assert v.n_states == 2
    assert v.state_names == ["P", "h"]

def test_component_algebraic_count():
    pipe = _AlgebraicPipe("pipe")
    assert pipe.n_states == 0
    assert pipe.n_algebraic == 1


def test_variable_registry_tracks_metadata_and_order():
    registry = VariableRegistry()
    registry.register(
        name="chamber.P",
        kind=VariableKind.STATE,
        units="Pa",
        scale=1e6,
        owner="chamber",
        description="Chamber pressure",
        bounds=(0.0, None),
    )
    registry.register(
        name="valve.command",
        kind=VariableKind.COMMAND,
        units="1",
        scale=1.0,
        owner="valve",
    )

    assert registry.names(VariableKind.STATE) == ["chamber.P"]
    assert registry.index("valve.command") == 1
    assert registry["chamber.P"].scale == 1e6
    assert registry["chamber.P"].bounds == (0.0, None)


def test_variable_registry_rejects_duplicates():
    registry = VariableRegistry()
    registry.register("chamber.P", VariableKind.STATE, units="Pa", scale=1e6)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("chamber.P", VariableKind.STATE, units="Pa", scale=1e6)


def test_residual_registry_tracks_scale_and_name_order():
    registry = ResidualRegistry()
    registry.register("pipe.flow_balance", units="kg/s", scale=10.0, owner="pipe")
    registry.register("connection.v1_to_v2.mdot", units="kg/s", scale=1.0)

    assert registry.names() == ["pipe.flow_balance", "connection.v1_to_v2.mdot"]
    assert registry.index("pipe.flow_balance") == 0
    assert registry["pipe.flow_balance"].scale == 10.0


def test_evaluation_result_normalizes_residuals_by_scale():
    result = EvaluationResult(
        dXdt=np.array([0.0]),
        Rz=np.array([10.0, -2.0]),
        outputs={"pipe.mdot": 1.0},
        residual_names=["r1", "r2"],
        output_names=["pipe.mdot"],
        residual_scales=np.array([5.0, 0.5]),
    )

    assert np.allclose(result.normalized_residuals, [2.0, -4.0])
    assert result.max_normalized_residual() == ("r2", -4.0)

def test_component_port_access():
    v = _SimpleVolume("vol")
    assert "inlet" in v.ports
    assert "outlet" in v.ports
    assert v.port("inlet").domain == PortDomain.FLUID

def test_component_port_missing_raises():
    v = _SimpleVolume("vol")
    with pytest.raises(KeyError, match="no port"):
        v.port("nonexistent")

def test_component_initial_state_values():
    v = _SimpleVolume("vol", initial_P=2e5, initial_h=5e5)
    assert v._state_values["P"] == 2e5
    assert v._state_values["h"] == 5e5


# ── Engine tests ──────────────────────────────────────────────────────────────

def test_engine_add_component():
    engine = Engine("test_engine")
    v = _SimpleVolume("vol1")
    engine.add_component(v)
    assert engine["vol1"] is v

def test_engine_duplicate_component_raises():
    engine = Engine("test_engine")
    v1 = _SimpleVolume("vol1")
    v2 = _SimpleVolume("vol1")  # same name
    engine.add_component(v1)
    with pytest.raises(ValueError, match="already added"):
        engine.add_component(v2)

def test_engine_compile_empty():
    engine = Engine("empty")
    layout = engine.compile()
    assert layout.n_states == 0
    assert layout.n_algebraic == 0

def test_engine_compile_state_count():
    engine = Engine("test")
    engine.add_component(_SimpleVolume("v1"))
    engine.add_component(_SimpleVolume("v2"))
    layout = engine.compile()
    assert layout.n_states == 4   # 2 states × 2 volumes

def test_engine_compile_alg_count():
    engine = Engine("test")
    engine.add_component(_AlgebraicPipe("p1"))
    engine.add_component(_AlgebraicPipe("p2"))
    layout = engine.compile()
    assert layout.n_algebraic == 2

def test_engine_assemble_scatter_roundtrip():
    engine = Engine("test")
    engine.add_component(_SimpleVolume("v1", initial_P=1e5, initial_h=3e5))
    engine.add_component(_SimpleVolume("v2", initial_P=2e5, initial_h=6e5))
    layout = engine.compile()
    X = layout.assemble_state_vector()
    assert len(X) == 4
    assert X[0] == 1e5   # v1.P
    assert X[1] == 3e5   # v1.h
    assert X[2] == 2e5   # v2.P
    assert X[3] == 6e5   # v2.h
    X_modified = X * 2
    layout.scatter_state_vector(X_modified)
    assert layout.components[0]._state_values["P"] == 2e5
    assert layout.components[1]._state_values["P"] == 4e5

def test_engine_all_state_names():
    engine = Engine("test")
    engine.add_component(_SimpleVolume("vol"))
    layout = engine.compile()
    names = layout.all_state_names()
    assert names == ["vol.P", "vol.h"]


def test_engine_compile_populates_variable_and_residual_registries():
    engine = Engine("test")
    engine.add_component(_SimpleVolume("vol"))
    engine.add_component(_AlgebraicPipe("pipe"))

    layout = engine.compile()

    assert layout.variable_registry.names(VariableKind.STATE) == ["vol.P", "vol.h"]
    assert layout.variable_registry.names(VariableKind.ALGEBRAIC) == ["pipe.flow_balance"]
    assert layout.residual_registry.names() == ["pipe.flow_balance"]
    assert layout.residual_registry["pipe.flow_balance"].owner == "pipe"


def test_layout_evaluate_returns_derivatives_residuals_and_outputs():
    engine = Engine("test")
    engine.add_component(_SimpleVolume("vol", initial_P=2e5, initial_h=5e5))
    engine.add_component(_AlgebraicPipe("pipe"))
    layout = engine.compile()

    result = layout.evaluate(
        t=0.0,
        X=layout.assemble_state_vector(),
        Z=np.array([0.0]),
        U={"mdot": 3.0, "flow_residual": -0.25},
    )

    assert isinstance(result, EvaluationResult)
    assert np.allclose(result.dXdt, [0.0, 0.0])
    assert np.allclose(result.Rz, [-0.25])
    assert result.residual_names == ["pipe.flow_balance"]
    assert result.outputs["vol.pressure"] == 2e5
    assert result.outputs["pipe.mdot"] == 3.0


def test_engine_compile_registers_fluid_connection_residuals():
    engine = Engine("test")
    pipe = _AlgebraicPipe("pipe")
    vol = _SimpleVolume("vol")
    engine.add_component(pipe)
    engine.add_component(vol)
    engine.connect(pipe.port("outlet"), vol.port("inlet"))

    layout = engine.compile()

    assert "connection.pipe.outlet__vol.inlet.P" in layout.residual_registry.names()
    assert "connection.pipe.outlet__vol.inlet.h" in layout.residual_registry.names()
    assert "connection.pipe.outlet__vol.inlet.mdot" in layout.residual_registry.names()


def test_layout_evaluate_reports_fluid_connection_mass_flow_residual():
    engine = Engine("test")
    pipe = _AlgebraicPipe("pipe")
    vol = _SimpleVolume("vol", initial_P=2e5, initial_h=5e5)
    engine.add_component(pipe)
    engine.add_component(vol)
    engine.connect(pipe.port("outlet"), vol.port("inlet"))
    layout = engine.compile()

    result = layout.evaluate(
        t=0.0,
        X=layout.assemble_state_vector(),
        Z=np.zeros(layout.n_algebraic),
        U={"pipe.mdot": 3.0, "vol.inlet.mdot": 2.5},
    )

    idx = result.residual_names.index("connection.pipe.outlet__vol.inlet.mdot")
    assert result.Rz[idx] == -0.5

def test_engine_connect_valid():
    engine = Engine("test")
    v1 = _SimpleVolume("v1")
    v2 = _SimpleVolume("v2")
    engine.add_component(v1).add_component(v2)
    engine.connect(v1.port("outlet"), v2.port("inlet"))
    assert v1.port("outlet").is_connected
    assert v2.port("inlet").is_connected

def test_engine_connect_domain_mismatch():
    from atha.core.port import PortConnectionError
    engine = Engine("test")
    v = _SimpleVolume("v")
    engine.add_component(v)

    class _RotorStub(BaseComponent):
        def __init__(self):
            super().__init__("rotor")
        def _declare_ports(self):
            self._register_port("shaft", ShaftPort("shaft", PortDirection.INLET, self))
        def _declare_states(self): pass
        def _declare_algebraic_vars(self): pass
        def compute_outputs(self, t, s, i): return {}
        def get_state_derivatives(self, t, s, i, o): return {}
        def get_residuals(self, t, s, i, o): return {}
        def initialize(self, op): pass

    rotor = _RotorStub()
    engine.add_component(rotor)
    with pytest.raises(PortConnectionError):
        engine.connect(v.port("outlet"), rotor.port("shaft"))
