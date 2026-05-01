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

    def compute_outputs(self, t, states, inputs): return {}
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

    def compute_outputs(self, t, states, inputs): return {}
    def get_state_derivatives(self, t, states, inputs, outputs): return {}
    def get_residuals(self, t, states, inputs, outputs): return {"flow_balance": 0.0}
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
