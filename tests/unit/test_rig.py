# tests/unit/test_rig.py
"""Unit tests for ComponentRig, ComponentSweep, SweepAxis, SweepResult, RigResult."""
import tempfile
import os
import numpy as np
import pytest

from atha.thermo.ideal_gas import IdealGasBackend
from atha.components.volume import Volume
from atha.components.valve import Valve
from atha.components.pump import Pump, PumpMap
from atha.analysis.component_rig import (
    ComponentRig, ComponentSweep, SweepAxis, SweepResult, RigResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gas():
    return IdealGasBackend(gamma=1.4, R=287.0)


def make_volume(name="vol", P0=1e5, T0=300.0):
    gas = make_gas()
    vol = Volume(name, volume=0.01, thermo=gas, initial_P=P0, initial_T=T0)
    vol.add_inlet("inlet")
    vol.add_outlet("outlet")
    return vol, gas


def make_valve(name="valve"):
    return Valve(name, max_area=1e-4, discharge_coeff=0.72)


def make_pump(name="pump"):
    gas = IdealGasBackend(gamma=1.4, R=287.0)
    pm = PumpMap.from_design_point(mdot_design=1.0, dP_design=1e6,
                                   speed_design=10000, efficiency_design=0.70)
    return Pump(name, diameter=0.10, pump_map=pm, fluid=gas), pm


# ---------------------------------------------------------------------------
# ComponentRig — algebraic component
# ---------------------------------------------------------------------------

class TestRigEvaluateAlgebraic:

    def test_valve_fully_open_returns_positive_flow(self):
        valve = make_valve()
        rig = ComponentRig(valve)
        out = rig.evaluate({
            "inlet.P": 2e5, "inlet.h": 0.0, "inlet.mdot": 0.0,
            "outlet.P": 1e5, "valve.A_frac": 1.0,
        })
        assert out["mdot"] > 0.0

    def test_valve_closed_zero_flow(self):
        valve = make_valve()
        rig = ComponentRig(valve)
        out = rig.evaluate({
            "inlet.P": 2e5, "inlet.h": 0.0, "inlet.mdot": 0.0,
            "outlet.P": 1e5, "valve.A_frac": 0.0,
        })
        assert out["mdot"] == pytest.approx(0.0, abs=1e-12)

    def test_pump_design_point_delta_P(self):
        pump, pm = make_pump()
        rig = ComponentRig(pump)
        out = rig.evaluate({
            "inlet.P": 5e5, "inlet.h": 1.0, "inlet.mdot": 1.0,
            "omega": pm.omega_design,
        })
        # At design omega, delta_P should match PumpMap design
        assert abs(out["delta_P"] - pm.dP_design) < 1.0

    def test_evaluate_returns_eta_alias(self):
        pump, pm = make_pump()
        rig = ComponentRig(pump)
        out = rig.evaluate({
            "inlet.P": 5e5, "inlet.h": 1.0, "inlet.mdot": 1.0,
            "omega": pm.omega_design,
        })
        assert "eta" in out


# ---------------------------------------------------------------------------
# ComponentRig — dynamic component (Volume)
# ---------------------------------------------------------------------------

class TestRigEvaluateDynamic:

    def test_volume_evaluate_returns_state_dict(self):
        vol, gas = make_volume(P0=2e5)
        rig = ComponentRig(vol)
        out = rig.evaluate({
            "inlet.mdot": 0.5, "inlet.h": gas.state_from_PT(2e5, 300.0).h,
            "outlet.mdot": 0.5,
        })
        assert "P" in out or "fluid_state" in out

    def test_volume_evaluate_converges_at_balanced_flow(self):
        vol, gas = make_volume(P0=1e5)
        rig = ComponentRig(vol)
        h0 = gas.state_from_PT(1e5, 300.0).h
        # Should not raise
        rig.evaluate({"inlet.mdot": 1.0, "inlet.h": h0, "outlet.mdot": 1.0})


# ---------------------------------------------------------------------------
# ComponentRig — transient
# ---------------------------------------------------------------------------

class TestRigTransient:

    def test_algebraic_transient_returns_rigresult(self):
        valve = make_valve()
        rig = ComponentRig(valve)

        def bcs_fn(t):
            return {"inlet.P": 2e5, "inlet.h": 0.0, "inlet.mdot": 0.0,
                    "outlet.P": 1e5, "valve.A_frac": 0.5}

        result = rig.transient((0.0, 0.5), bcs_fn, recording_rate_hz=50.0)
        assert isinstance(result, RigResult)
        assert len(result.t) > 1
        assert result.X is None
        assert len(result.state_names) == 0

    def test_algebraic_transient_length_matches_rate(self):
        valve = make_valve()
        rig = ComponentRig(valve)
        bcs_fn = lambda t: {"inlet.P": 2e5, "inlet.h": 0.0,
                            "inlet.mdot": 0.0, "outlet.P": 1e5, "valve.A_frac": 1.0}
        result = rig.transient((0.0, 1.0), bcs_fn, recording_rate_hz=100.0)
        assert len(result.t) == 101  # 0..1s at 100 Hz = 101 points

    def test_dynamic_transient_pressure_rises_with_inflow(self):
        vol, gas = make_gas(), make_gas()
        gas = make_gas()
        vol = Volume("v", volume=0.01, thermo=gas, initial_P=1e5, initial_T=300.0)
        vol.add_inlet("inlet")
        rig = ComponentRig(vol)
        h0 = gas.state_from_PT(1e5, 300.0).h

        def bcs_fn(t):
            return {"inlet.mdot": 0.05, "inlet.h": h0}

        result = rig.transient((0.0, 2.0), bcs_fn, recording_rate_hz=50.0)
        assert isinstance(result, RigResult)
        assert result.X is not None
        assert result.X.shape[1] == vol.n_states
        # Pressure state should rise over time
        P_vals = result.X[:, 0]  # P is first state
        assert P_vals[-1] > P_vals[0]


# ---------------------------------------------------------------------------
# RigResult
# ---------------------------------------------------------------------------

class TestRigResult:

    def _make_result(self):
        t = np.linspace(0, 1, 11)
        outputs = {"flow": np.linspace(0, 1, 11), "pressure": np.ones(11) * 2e5}
        return RigResult("test_comp", t, outputs, state_names=[], X=None)

    def test_get_known_key(self):
        r = self._make_result()
        arr = r.get("flow")
        assert arr.shape == (11,)

    def test_get_missing_raises_keyerror(self):
        r = self._make_result()
        with pytest.raises(KeyError, match="flow"):  # message mentions available keys
            r.get("nonexistent_key_xyz")

    def test_save_load_roundtrip(self, tmp_path):
        r = self._make_result()
        path = str(tmp_path / "rig_result.h5")
        r.save(path)
        loaded = RigResult.load(path)
        assert loaded.component_name == r.component_name
        np.testing.assert_allclose(loaded.t, r.t)
        np.testing.assert_allclose(loaded.outputs["flow"], r.outputs["flow"])


# ---------------------------------------------------------------------------
# ComponentSweep
# ---------------------------------------------------------------------------

class TestComponentSweep:

    def _make_pump_rig(self):
        pump, pm = make_pump()
        rig = ComponentRig(pump)
        return rig, pm

    def test_1d_sweep_shape(self):
        rig, pm = self._make_pump_rig()
        sweep = ComponentSweep(
            rig=rig,
            axes=[SweepAxis("omega", np.linspace(5000, 15000, 8) * np.pi / 30)],
            fixed_bcs={"inlet.P": 5e5, "inlet.h": 1.0, "inlet.mdot": 1.0},
            outputs=["delta_P", "eta"],
        )
        result = sweep.run()
        assert result.get("delta_P").shape == (8,)
        assert result.n_failed == 0

    def test_2d_sweep_shape(self):
        rig, pm = self._make_pump_rig()
        sweep = ComponentSweep(
            rig=rig,
            axes=[
                SweepAxis("omega",      np.linspace(5000, 15000, 5) * np.pi / 30),
                SweepAxis("inlet.mdot", np.linspace(0.5, 2.0, 4)),
            ],
            fixed_bcs={"inlet.P": 5e5, "inlet.h": 1.0},
            outputs=["delta_P"],
        )
        result = sweep.run()
        assert result.get("delta_P").shape == (5, 4)

    def test_failed_points_recorded_as_nan(self):
        """Force a failure by providing an impossible BCS that triggers an exception."""
        valve = make_valve()
        rig = ComponentRig(valve)

        # Monkey-patch evaluate to raise on one specific position value
        original_evaluate = rig.evaluate
        def patched(bcs):
            if bcs.get("valve.A_frac", 1.0) < 0.0:
                raise ValueError("forced failure")
            return original_evaluate(bcs)
        rig.evaluate = patched

        sweep = ComponentSweep(
            rig=rig,
            axes=[SweepAxis("valve.A_frac", np.array([-0.1, 0.5, 1.0]))],
            fixed_bcs={"inlet.P": 2e5, "inlet.h": 0.0,
                       "inlet.mdot": 0.0, "outlet.P": 1e5},
            outputs=["mdot"],
        )
        result = sweep.run()
        assert result.n_failed == 1
        assert np.isnan(result.get("mdot")[0])
        assert not np.isnan(result.get("mdot")[1])

    def test_1axis_plot_map_raises(self):
        rig, _ = self._make_pump_rig()
        sweep = ComponentSweep(
            rig=rig,
            axes=[SweepAxis("omega", np.linspace(5000, 15000, 4) * np.pi / 30)],
            fixed_bcs={"inlet.P": 5e5, "inlet.h": 1.0, "inlet.mdot": 1.0},
            outputs=["delta_P"],
        )
        result = sweep.run()
        with pytest.raises(ValueError):
            result.plot_map("delta_P", x_axis="omega", y_axis="omega", show=False)


# ---------------------------------------------------------------------------
# SweepResult save / load
# ---------------------------------------------------------------------------

class TestSweepResultIO:

    def test_save_load_roundtrip(self, tmp_path):
        axes = {"omega": np.array([1.0, 2.0, 3.0])}
        outputs = {"eta": np.array([0.6, 0.7, 0.65])}
        failed = np.array([False, False, False])
        sr = SweepResult(axes, outputs, failed)
        path = str(tmp_path / "sweep.h5")
        sr.save(path)
        loaded = SweepResult.load(path)
        np.testing.assert_allclose(loaded.axes["omega"], axes["omega"])
        np.testing.assert_allclose(loaded.outputs["eta"], outputs["eta"])
        assert loaded.n_failed == 0


# ---------------------------------------------------------------------------
# SweepResult plot_curve
# ---------------------------------------------------------------------------

class TestSweepResultPlotCurve:

    def test_plot_curve_1d(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        axes = {"omega": np.linspace(1, 5, 5)}
        outputs = {"dP": np.linspace(100, 500, 5)}
        sr = SweepResult(axes, outputs, np.zeros(5, dtype=bool))
        fig = sr.plot_curve("dP", sweep_axis="omega", show=False)
        assert fig is not None

    def test_plot_curve_2d_with_fixed(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        omega_vals = np.array([1.0, 2.0, 3.0])
        mdot_vals  = np.array([0.5, 1.0])
        dP_grid = np.outer(omega_vals, mdot_vals)  # shape (3, 2)
        sr = SweepResult(
            axes={"omega": omega_vals, "mdot": mdot_vals},
            outputs={"dP": dP_grid},
            failed_mask=np.zeros((3, 2), dtype=bool),
        )
        fig = sr.plot_curve("dP", sweep_axis="omega",
                            fixed={"mdot": 0.5}, show=False)
        assert fig is not None

    def test_plot_curve_missing_axis_raises(self):
        sr = SweepResult(
            axes={"omega": np.array([1.0, 2.0])},
            outputs={"dP": np.array([100.0, 200.0])},
            failed_mask=np.zeros(2, dtype=bool),
        )
        with pytest.raises(KeyError):
            sr.plot_curve("dP", sweep_axis="nonexistent", show=False)


# ---------------------------------------------------------------------------
# required_inputs
# ---------------------------------------------------------------------------

class TestRequiredInputs:

    def test_pump_required_inputs(self):
        pump, _ = make_pump()
        rig = ComponentRig(pump)
        keys = rig.required_inputs()
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert "inlet.P" in keys

    def test_volume_required_inputs(self):
        vol, _ = make_volume()
        rig = ComponentRig(vol)
        keys = rig.required_inputs()
        assert isinstance(keys, list)
