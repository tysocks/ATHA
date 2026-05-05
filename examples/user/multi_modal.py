from atha.profiles import (
    PhaseDefinition, PhaseMode, ControlCommand,
    SafetyLimit, TestProfile,
)
from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.core.engine import Engine

# --- build engine ---
gas = IdealGasBackend(gamma=1.4, R=287.0)
vol = Volume("chamber", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
vol.add_inlet("inlet")
engine = Engine("e")
engine.add_component(vol)
layout = engine.compile()
X0 = layout.assemble_state_vector()
h_ref = gas.state_from_PT(1e5, 300.0).h

# --- define profile ---
profile = TestProfile(
    name="fill_sequence",
    phases=[
        # Phase 1: trim to steady state with no inflow
        PhaseDefinition(
            name="pre_fill_trim",
            mode=PhaseMode.STEADY_TRIM,
            duration=5.0,
            trim_targets={"inlet.mdot": 0.0},
        ),
        # Phase 2: open valve ramp — mdot rises linearly 0→0.1 kg/s over 2 s
        PhaseDefinition(
            name="fill",
            mode=PhaseMode.TRANSIENT,
            duration=2.0,
            control_commands=[
                ControlCommand("inlet.mdot", fn=lambda t: 0.05 * t),
                ControlCommand("inlet.h",   fn=lambda t: h_ref),
            ],
            recording_rate_hz=100.0,   # sample rate for stored time series
        ),
        # Phase 3: hold at end state
        PhaseDefinition(
            name="hold",
            mode=PhaseMode.DWELL,
            duration=1.0,
        ),
    ],
    global_limits=[
        # Hard limit: abort if chamber pressure exceeds 2 bar
        SafetyLimit("P_max", component_name="chamber", state_name="P",
                    upper_limit=20e5, is_hard=True),
    ],
)

result = profile.execute(layout, X0)
if result.success:
    print(f"Total duration: {result.total_duration:.2f} s")

    # Access a single phase
    fill = result.get_phase("fill")
    P = fill.get("chamber", "P")   # numpy array at recorded time points
    print(f"Peak pressure: {P.max()/1e5:.2f} bar")

    # Stitch chamber pressure across all phases (global time, one state)
    t_all, P_all = result.get_combined("chamber", "P")
    print(f"Full-profile P range: {P_all.min()/1e5:.2f} – {P_all.max()/1e5:.2f} bar")

    # Quick matplotlib overview of every state across all phases
    result.plot_timeline()

else:
    print(f"Abort: {result.abort_reason} at t={result.abort_time:.3f} s")