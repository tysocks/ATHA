# tests/unit/test_profiles.py
from atha.profiles.phase import PhaseDefinition, PhaseMode, ControlCommand
from atha.profiles.limits import SafetyLimit, AbortManager, EngineAbort
from atha.profiles.result import PhaseResult, TestProfileResult
from atha.profiles.executor import execute_phase
from atha.profiles.profile import TestProfile
from atha.profiles.io import save_profile_result, load_profile_result
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
import numpy as np
import pytest
import tempfile
import os


def test_phase_mode_values():
    assert PhaseMode.STEADY_TRIM.value == "steady_trim"
    assert PhaseMode.TRANSIENT.value == "transient"
    assert PhaseMode.DWELL.value == "dwell"


def test_control_command_evaluates():
    cmd = ControlCommand(bcs_key="throttle.cmd", fn=lambda t: 1.0 - 0.35 * t / 5.0)
    assert abs(cmd.fn(0.0) - 1.0) < 1e-10
    assert abs(cmd.fn(5.0) - 0.65) < 1e-10


def test_phase_definition_defaults():
    phase = PhaseDefinition(
        name="mainstage",
        mode=PhaseMode.STEADY_TRIM,
        duration=30.0,
    )
    assert phase.name == "mainstage"
    assert phase.mode == PhaseMode.STEADY_TRIM
    assert phase.trim_targets == {}
    assert phase.control_commands == []
    assert phase.abort_checks == []
    assert phase.recording_rate_hz == 100.0


def test_phase_definition_with_commands():
    cmd = ControlCommand("throttle.cmd", fn=lambda t: 0.65)
    phase = PhaseDefinition(
        name="transient",
        mode=PhaseMode.TRANSIENT,
        duration=5.0,
        control_commands=[cmd],
        recording_rate_hz=200.0,
    )
    assert len(phase.control_commands) == 1
    assert phase.control_commands[0].bcs_key == "throttle.cmd"


def test_safety_limit_upper_violation():
    limit = SafetyLimit(
        name="Pc_max", component_name="chamber", state_name="P",
        upper_limit=25e6, is_hard=True,
    )
    assert limit.upper_limit == 25e6
    assert limit.is_hard is True
    assert limit.lower_limit is None


def test_engine_abort_exception():
    exc = EngineAbort(reason="Pc exceeded 25 MPa", t=3.52)
    assert exc.reason == "Pc exceeded 25 MPa"
    assert exc.t == 3.52
    assert "Pc exceeded" in str(exc)


def test_abort_manager_no_violation():
    limit = SafetyLimit("Pc_max", "chamber", "P", upper_limit=25e6, is_hard=True)
    mgr = AbortManager(limits=[limit])

    class MockComp:
        name = "chamber"
        state_names = ["P", "h"]
        _state_values = {"P": 10e6, "h": 1e6}

    class MockLayout:
        components = [MockComp()]
        state_offsets = {"chamber": 0}

    X = np.array([10e6, 1e6])
    # Should not raise
    mgr.check(MockLayout(), X)


def test_abort_manager_hard_violation_raises():
    limit = SafetyLimit("Pc_max", "chamber", "P", upper_limit=25e6, is_hard=True)
    mgr = AbortManager(limits=[limit])

    class MockComp:
        name = "chamber"
        state_names = ["P", "h"]
        _state_values = {"P": 26e6, "h": 1e6}

    class MockLayout:
        components = [MockComp()]
        state_offsets = {"chamber": 0}

    X = np.array([26e6, 1e6])
    with pytest.raises(EngineAbort, match="Pc_max"):
        mgr.check(MockLayout(), X)


def test_phase_result_get_state():
    t = np.array([0.0, 0.5, 1.0])
    X = np.array([[1e5, 3e5], [1.1e5, 3e5], [1.2e5, 3e5]])
    pr = PhaseResult(
        name="test_phase",
        t=t, X=X,
        state_names=["vol.P", "vol.h"],
        X_final=X[-1],
        abort_triggered=False,
    )
    P = pr.get("vol", "P")
    np.testing.assert_array_almost_equal(P, [1e5, 1.1e5, 1.2e5])


def test_phase_result_get_missing_raises():
    pr = PhaseResult("p", np.array([0.0]), np.array([[1.0]]),
                     ["vol.P"], np.array([1.0]), False)
    with pytest.raises(KeyError):
        pr.get("vol", "omega")


def test_test_profile_result_duration():
    pr1 = PhaseResult("a", np.array([0.0, 1.0]), np.zeros((2, 1)),
                      ["v.P"], np.zeros(1), False)
    pr2 = PhaseResult("b", np.array([0.0, 2.0]), np.zeros((2, 1)),
                      ["v.P"], np.zeros(1), False)
    result = TestProfileResult(
        profile_name="test", phases=[pr1, pr2],
        state_names=["v.P"],
    )
    assert result.total_duration == pytest.approx(3.0)
    assert result.abort_reason is None


def test_test_profile_result_abort():
    result = TestProfileResult(
        profile_name="test", phases=[],
        state_names=[],
        abort_reason="Pc exceeded limit",
        abort_time=2.5,
    )
    assert result.abort_reason == "Pc exceeded limit"
    assert result.success is False


def test_downsample_uniform_rate():
    """Mock a TransientSolution and downsample it at a fixed rate."""
    from atha.profiles.recording import downsample_dense_output
    from atha.solver.transient import TransientSolution

    # Simulate a linear state rising from 1e5 to 2e5 Pa over 2 seconds
    t_integrator = np.array([0.0, 0.27, 0.61, 1.05, 1.58, 2.0])
    P_values = 1e5 + (t_integrator / 2.0) * 1e5
    X = P_values.reshape(-1, 1)
    sol = TransientSolution(t=t_integrator, X=X, state_names=["vol.P"])

    recording_rate_hz = 10.0
    t_duration = 2.0
    result = downsample_dense_output(sol, t_duration, recording_rate_hz)

    assert len(result.t) == 21         # 0, 0.1, 0.2, ..., 2.0
    assert result.t[0] == pytest.approx(0.0)
    assert result.t[-1] == pytest.approx(2.0)
    # P should still be monotonically rising
    assert np.all(np.diff(result.get("vol", "P")) >= 0)


def test_downsample_uses_interpolation():
    """Downsampled values should lie between adjacent integrator values."""
    from atha.profiles.recording import downsample_dense_output
    from atha.solver.transient import TransientSolution

    t_raw = np.array([0.0, 1.0])
    X_raw = np.array([[0.0], [10.0]])
    sol = TransientSolution(t=t_raw, X=X_raw, state_names=["r.omega"])

    result = downsample_dense_output(sol, 1.0, recording_rate_hz=10.0)
    omega = result.get("r", "omega")

    # At t=0.5s, interpolated value should be 5.0
    idx_half = np.argmin(np.abs(result.t - 0.5))
    assert omega[idx_half] == pytest.approx(5.0, rel=0.01)


# ── executor tests ─────────────────────────────────────────────────────────────

def _make_layout():
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    vol = Volume("vol", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
    vol.add_inlet("inlet")
    engine = Engine("e")
    engine.add_component(vol)
    layout = engine.compile()
    return layout, gas


def test_execute_steady_trim_phase():
    """STEADY_TRIM phase: solver runs and returns a PhaseResult."""
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()

    phase = PhaseDefinition(
        name="trim",
        mode=PhaseMode.STEADY_TRIM,
        duration=10.0,
        trim_targets={
            "inlet.mdot": 0.0,
        },
    )
    result = execute_phase(layout, X0, phase, global_limits=[])
    assert result.name == "trim"
    assert len(result.X_final) == len(X0)


def test_execute_transient_phase_pressure_rises():
    """TRANSIENT phase: constant inflow raises pressure."""
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    phase = PhaseDefinition(
        name="fill",
        mode=PhaseMode.TRANSIENT,
        duration=2.0,
        control_commands=[
            ControlCommand("inlet.mdot", fn=lambda t: 0.05),
            ControlCommand("inlet.h",   fn=lambda t: h_ref),
        ],
        recording_rate_hz=10.0,
    )
    result = execute_phase(layout, X0, phase, global_limits=[])
    P_series = result.get("vol", "P")
    assert P_series[-1] > P_series[0], "Pressure should rise with constant inflow"


def test_execute_transient_abort_on_limit():
    """TRANSIENT phase: hard limit triggers EngineAbort."""
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    phase = PhaseDefinition(
        name="fill",
        mode=PhaseMode.TRANSIENT,
        duration=10.0,
        control_commands=[
            ControlCommand("inlet.mdot", fn=lambda t: 0.2),
            ControlCommand("inlet.h",   fn=lambda t: h_ref),
        ],
    )
    limit = SafetyLimit("P_max", "vol", "P", upper_limit=1.2e5, is_hard=True)

    with pytest.raises(EngineAbort):
        execute_phase(layout, X0, phase, global_limits=[limit])


def test_two_phase_profile_state_handoff():
    """Steady trim then transient: X_final of phase 1 becomes X0 of phase 2."""
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    profile = TestProfile(
        name="two_phase",
        phases=[
            PhaseDefinition(
                name="trim",
                mode=PhaseMode.STEADY_TRIM,
                duration=5.0,
                trim_targets={"inlet.mdot": 0.0},
            ),
            PhaseDefinition(
                name="fill",
                mode=PhaseMode.TRANSIENT,
                duration=1.0,
                control_commands=[
                    ControlCommand("inlet.mdot", fn=lambda t: 0.05),
                    ControlCommand("inlet.h",   fn=lambda t: h_ref),
                ],
                recording_rate_hz=10.0,
            ),
        ],
    )

    result = profile.execute(layout, X0)
    assert result.success
    assert len(result.phases) == 2
    assert result.phases[0].name == "trim"
    assert result.phases[1].name == "fill"
    # Pressure in fill phase should rise
    P_fill = result.phases[1].get("vol", "P")
    assert P_fill[-1] > P_fill[0]


def test_profile_aborts_on_hard_limit():
    layout, gas = _make_layout()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(1e5, 300.0).h

    profile = TestProfile(
        name="abort_test",
        phases=[
            PhaseDefinition(
                name="overfill",
                mode=PhaseMode.TRANSIENT,
                duration=10.0,
                control_commands=[
                    ControlCommand("inlet.mdot", fn=lambda t: 0.3),
                    ControlCommand("inlet.h",   fn=lambda t: h_ref),
                ],
            ),
        ],
        global_limits=[
            SafetyLimit("P_max", "vol", "P", upper_limit=1.1e5, is_hard=True),
        ],
    )

    result = profile.execute(layout, X0)
    assert not result.success
    assert result.abort_reason is not None
    assert result.abort_time is not None


def test_profile_result_roundtrip_hdf5():
    """Save and reload a TestProfileResult; verify data matches."""
    t = np.linspace(0, 1.0, 11)
    X = np.column_stack([1e5 + t * 1e4, 3e5 * np.ones_like(t)])
    pr = PhaseResult("fill", t, X, ["vol.P", "vol.h"], X[-1], False)
    result = TestProfileResult("test", [pr], ["vol.P", "vol.h"])

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        fname = f.name

    try:
        save_profile_result(result, fname)
        loaded = load_profile_result(fname)

        assert loaded.profile_name == "test"
        assert len(loaded.phases) == 1
        assert loaded.phases[0].name == "fill"
        np.testing.assert_array_almost_equal(
            loaded.phases[0].get("vol", "P"),
            result.phases[0].get("vol", "P"),
        )
    finally:
        os.unlink(fname)
