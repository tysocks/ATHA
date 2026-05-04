# tests/integration/test_profile_ttbe.py
"""Integration test: Test profile execution using a simple Volume-based engine."""
import numpy as np
import pytest
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine
from atha.profiles import (
    TestProfile, PhaseDefinition, PhaseMode,
    ControlCommand, SafetyLimit,
)


def build_simple_engine():
    """Minimal engine: single volume representing main chamber."""
    gas = IdealGasBackend(gamma=1.24, R=711.0)
    chamber = Volume(
        "chamber", volume=0.05, thermo=gas,
        initial_P=20.6e6, initial_T=3560.0,
    )
    chamber.add_inlet("propellant_in")
    chamber.add_outlet("nozzle_out")
    engine = Engine("ttbe")
    engine.add_component(chamber)
    return engine, gas


def test_ttbe_throttle_sweep_three_trim_points():
    """Execute 100%→65%→100% profile; all phases complete successfully."""
    engine, gas = build_simple_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()

    h_ref = gas.state_from_PT(20.6e6, 3560.0).h
    mdot_100 = 468.0
    mdot_65  = 468.0 * 0.65

    profile = TestProfile(
        name="ttbe_throttle_sweep",
        phases=[
            PhaseDefinition(
                name="mainstage_100pct",
                mode=PhaseMode.STEADY_TRIM,
                duration=30.0,
                trim_targets={
                    "propellant_in.mdot": mdot_100,
                    "propellant_in.h": h_ref,
                    "nozzle_out.mdot": mdot_100,
                },
            ),
            PhaseDefinition(
                name="throttle_down",
                mode=PhaseMode.TRANSIENT,
                duration=5.0,
                control_commands=[
                    ControlCommand("propellant_in.mdot",
                                   fn=lambda t, _m100=mdot_100, _m65=mdot_65:
                                       _m100 - (_m100 - _m65) * t / 5.0),
                    ControlCommand("propellant_in.h", fn=lambda t, _h=h_ref: _h),
                    ControlCommand("nozzle_out.mdot",
                                   fn=lambda t, _m100=mdot_100, _m65=mdot_65:
                                       _m100 - (_m100 - _m65) * t / 5.0),
                ],
                recording_rate_hz=20.0,
            ),
            PhaseDefinition(
                name="mainstage_65pct",
                mode=PhaseMode.STEADY_TRIM,
                duration=30.0,
                trim_targets={
                    "propellant_in.mdot": mdot_65,
                    "propellant_in.h": h_ref,
                    "nozzle_out.mdot": mdot_65,
                },
            ),
        ],
        global_limits=[
            SafetyLimit("Pc_hard_max", "chamber", "P",
                        upper_limit=30e6, is_hard=True),
        ],
    )

    result = profile.execute(layout, X0)

    assert result.success, f"Profile aborted: {result.abort_reason}"
    assert len(result.phases) == 3
    assert result.phases[0].name == "mainstage_100pct"
    assert result.phases[2].name == "mainstage_65pct"


def test_ttbe_profile_abort_on_overpressure():
    """Verify hard limit abort fires correctly."""
    engine, gas = build_simple_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()
    h_ref = gas.state_from_PT(20.6e6, 3560.0).h

    profile = TestProfile(
        name="abort_test",
        phases=[
            PhaseDefinition(
                name="overfill",
                mode=PhaseMode.TRANSIENT,
                duration=10.0,
                control_commands=[
                    ControlCommand("propellant_in.mdot", fn=lambda t: 2000.0),
                    ControlCommand("propellant_in.h", fn=lambda t, _h=h_ref: _h),
                    ControlCommand("nozzle_out.mdot", fn=lambda t: 0.0),
                ],
            ),
        ],
        global_limits=[
            SafetyLimit("Pc_limit", "chamber", "P", upper_limit=21e6, is_hard=True),
        ],
    )

    result = profile.execute(layout, X0)
    assert not result.success
    assert result.abort_reason is not None


def test_profile_hdf5_roundtrip(tmp_path):
    from atha.profiles.io import save_profile_result, load_profile_result
    engine, gas = build_simple_engine()
    layout = engine.compile()
    X0 = layout.assemble_state_vector()

    profile = TestProfile(
        name="roundtrip",
        phases=[
            PhaseDefinition(
                name="dwell",
                mode=PhaseMode.DWELL,
                duration=1.0,
            ),
        ],
    )
    result = profile.execute(layout, X0)
    fname = str(tmp_path / "test.hdf5")
    save_profile_result(result, fname)
    loaded = load_profile_result(fname)
    assert loaded.profile_name == "roundtrip"
    assert loaded.success
