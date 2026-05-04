# tests/unit/test_performance_map.py
import math
import numpy as np
import pytest
from atha.maps.performance_map import PerformanceMap


# ── from_arrays ──────────────────────────────────────────────────────────────

def test_1d_array_map_exact_at_nodes():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = x ** 2
    pm = PerformanceMap.from_arrays({"x": x}, {"y": y})
    result = pm(x=2.0)
    assert abs(result["y"] - 4.0) < 1e-10


def test_1d_array_map_interpolates():
    x = np.linspace(0, 10, 100)
    pm = PerformanceMap.from_arrays({"x": x}, {"y": x * 2.5})
    result = pm(x=5.0)
    assert abs(result["y"] - 12.5) < 1e-8


def test_2d_array_map():
    MR = np.array([2.0, 3.0, 4.0])
    Pc = np.array([1e6, 2e6, 3e6])
    # eta = 0.9 + 0.01*MR - 1e-8*Pc (fake surface)
    eta_grid = np.array([[0.9 + 0.01*mr - 1e-8*pc for pc in Pc] for mr in MR])
    pm = PerformanceMap.from_arrays({"MR": MR, "Pc": Pc}, {"eta": eta_grid})
    result = pm(MR=3.0, Pc=2e6)
    expected = 0.9 + 0.01*3.0 - 1e-8*2e6
    assert abs(result["eta"] - expected) < 1e-6


# ── from_scattered ────────────────────────────────────────────────────────────

def test_scattered_map_recovers_linear_surface():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(0, 5, n)
    y = rng.uniform(0, 5, n)
    z = 2.0 * x + 3.0 * y  # linear → RBF should interpolate exactly at nodes
    pm = PerformanceMap.from_scattered({"x": x, "y": y}, {"z": z})
    for i in range(5):
        result = pm.evaluate({"x": x[i], "y": y[i]})
        assert abs(result["z"] - z[i]) < 0.1  # RBF may have small error at nodes


# ── from_callable ─────────────────────────────────────────────────────────────

def test_callable_map():
    pm = PerformanceMap.from_callable(
        fn=lambda MR, Pc: {"eta_cstar": 0.9 + 0.01 * MR},
        axes=["MR", "Pc"],
        outputs=["eta_cstar"],
    )
    result = pm(MR=2.5, Pc=1e6)
    assert abs(result["eta_cstar"] - 0.925) < 1e-12


def test_callable_map_with_dotted_axis():
    pm = PerformanceMap.from_callable(
        fn=lambda **kw: {"Cd": 0.70 + 0.02 * kw["inlet.P"] / 3e5},
        axes=["inlet.P"],
        outputs=["Cd"],
    )
    result = pm.evaluate({"inlet.P": 3e5, "other_key": 99})
    assert abs(result["Cd"] - 0.72) < 1e-10


# ── constant ──────────────────────────────────────────────────────────────────

def test_constant_map():
    pm = PerformanceMap.constant(eta_cstar=0.975, Cd=0.72)
    result = pm()
    assert result["eta_cstar"] == 0.975
    assert result["Cd"] == 0.72


def test_constant_map_ignores_context():
    pm = PerformanceMap.constant(eta_cstar=0.975)
    result = pm.evaluate({"inlet.P": 3e5, "omega": 20000})
    assert result["eta_cstar"] == 0.975


# ── evaluate (context extraction) ────────────────────────────────────────────

def test_evaluate_ignores_extra_keys():
    pm = PerformanceMap.from_arrays({"x": np.linspace(0, 1, 10)},
                                    {"y": np.linspace(0, 1, 10)})
    result = pm.evaluate({"x": 0.5, "completely_irrelevant": 99999})
    assert abs(result["y"] - 0.5) < 1e-8


def test_evaluate_raises_on_missing_axis():
    pm = PerformanceMap.from_arrays({"x": np.linspace(0, 1, 5)},
                                    {"y": np.linspace(0, 1, 5)})
    with pytest.raises(KeyError, match="axis"):
        pm.evaluate({"z": 0.5})


# ── bounds / extrapolation ────────────────────────────────────────────────────

def test_clamp_extrapolation_silent():
    x = np.array([0.0, 1.0, 2.0])
    pm = PerformanceMap.from_arrays({"x": x}, {"y": x}, extrapolation="clamp")
    result = pm(x=5.0)   # beyond max, should clamp to 2.0
    assert result["y"] == pytest.approx(2.0)


def test_warn_extrapolation_emits_warning():
    x = np.array([0.0, 1.0, 2.0])
    pm = PerformanceMap.from_arrays({"x": x}, {"y": x}, extrapolation="warn")
    with pytest.warns(UserWarning, match="outside data bounds"):
        pm(x=5.0)


def test_error_extrapolation_raises():
    x = np.array([0.0, 1.0, 2.0])
    pm = PerformanceMap.from_arrays({"x": x}, {"y": x}, extrapolation="error")
    with pytest.raises(ValueError, match="outside data bounds"):
        pm(x=5.0)


# ── rasterize ─────────────────────────────────────────────────────────────────

def test_rasterize_callable_to_arrays():
    pm_fn = PerformanceMap.from_callable(
        fn=lambda x: {"y": x ** 2},
        axes=["x"],
        outputs=["y"],
    )
    grid = {"x": np.linspace(0, 3, 20)}
    pm_arr = pm_fn.rasterize(grid)
    assert pm_arr._source == "arrays"
    result = pm_arr(x=2.0)
    assert abs(result["y"] - 4.0) < 0.1


# ── HDF5 round-trip ───────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    x = np.linspace(0, 5, 50)
    pm = PerformanceMap.from_arrays({"x": x}, {"y": x * 2.0})
    path = str(tmp_path / "test_map.h5")
    pm.save(path)
    pm2 = PerformanceMap.load(path)
    assert pm2._source == "arrays"
    result = pm2(x=2.5)
    assert abs(result["y"] - 5.0) < 1e-8


def test_save_constant_roundtrip(tmp_path):
    pm = PerformanceMap.constant(eta=0.97, Cd=0.72)
    path = str(tmp_path / "const.h5")
    pm.save(path)
    pm2 = PerformanceMap.load(path)
    assert pm2()["eta"] == pytest.approx(0.97)
    assert pm2()["Cd"] == pytest.approx(0.72)


def test_callable_save_raises():
    pm = PerformanceMap.from_callable(
        fn=lambda x: {"y": x},
        axes=["x"],
        outputs=["y"],
    )
    with pytest.raises(ValueError, match="callable"):
        pm.save("nowhere.h5")


# ── component integration ─────────────────────────────────────────────────────

def test_pump_with_efficiency_map():
    from atha.components.pump import Pump, PumpMap
    from atha.thermo.ideal_gas import IdealGasBackend

    # Map: efficiency drops linearly with mdot
    mdot_vals = np.array([0.5, 1.0, 1.5, 2.0])
    eta_vals  = np.array([0.80, 0.75, 0.70, 0.65])
    eff_map = PerformanceMap.from_arrays(
        {"inlet.mdot": mdot_vals},
        {"efficiency": eta_vals},
    )
    pm = PumpMap(mdot_design=1.0, dP_design=1e5,
                 omega_design=1000.0 * math.pi / 30.0, eta_design=0.75)
    fluid = IdealGasBackend(gamma=1.4, R=287.0)
    pump = Pump("p", diameter=0.1, pump_map=pm, fluid=fluid,
                efficiency_map=eff_map)
    h_in = fluid.cp * 300.0
    out = pump.compute_outputs(0.0, {}, {"shaft.omega": 1000.0 * math.pi / 30.0,
                                          "inlet.mdot": 1.5,
                                          "inlet.P": 1e5,
                                          "inlet.h": h_in})
    assert abs(out["efficiency"] - 0.70) < 1e-6


def test_orifice_with_cda_map():
    from atha.components.orifice import OrificeCompressible

    # CdA increases with upstream pressure (bellows-like)
    P_vals  = np.array([1e5, 2e5, 3e5, 4e5])
    CdA_vals = np.array([1e-4, 1.5e-4, 1.8e-4, 2e-4])
    cda_map = PerformanceMap.from_arrays(
        {"inlet.P": P_vals},
        {"CdA": CdA_vals},
    )
    orifice = OrificeCompressible("o", area=1e-4, cda_map=cda_map)
    out1 = orifice.compute_outputs(0.0, {}, {"inlet.P": 1e5, "outlet.P": 0.5e5,
                                               "inlet.T": 300.0})
    out2 = orifice.compute_outputs(0.0, {}, {"inlet.P": 4e5, "outlet.P": 0.5e5,
                                               "inlet.T": 300.0})
    # Higher pressure → larger CdA → more flow
    assert out2["mdot"] > out1["mdot"]


def test_valve_with_cd_map():
    from atha.components.valve import Valve

    # Cd varies with pressure ratio
    dp_vals = np.array([1e3, 1e4, 1e5])
    cd_vals  = np.array([0.6, 0.7, 0.8])
    cd_map = PerformanceMap.from_arrays({"valve.A_frac": np.linspace(0, 1, 3)},
                                         {"Cd": cd_vals})
    valve = Valve("v", max_area=1e-3, cd_map=cd_map)
    out = valve.compute_outputs(0.0, {}, {
        "inlet.P": 2e5, "outlet.P": 1e5,
        "inlet.rho": 1000.0, "valve.A_frac": 0.5,
    })
    assert out["mdot"] > 0.0
