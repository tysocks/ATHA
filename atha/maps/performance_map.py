# atha/maps/performance_map.py
from __future__ import annotations

import json
import warnings
from typing import Callable, Dict, List, Optional

import h5py
import numpy as np


class PerformanceMap:
    """
    General-purpose multi-axis performance map.

    Axes can be ANY named quantities: states, inputs, computed outputs, or
    cross-component values passed through BCS. The map extracts its declared
    axis values from a context dict by name at evaluation time.

    Sources
    -------
    from_arrays    – structured (Cartesian product) grid of numpy arrays
    from_scattered – arbitrary scattered data points (RBF interpolation)
    from_callable  – any Python callable fn(**axis_values) -> dict
    constant       – fixed values regardless of operating conditions

    Extrapolation modes: "clamp" (default), "warn", "error"

    Usage
    -----
    Keyword convenience::

        map(inlet.P=3e5, position=0.7)   # doesn't work — dots in names

    Use evaluate() for axis names that contain dots::

        map.evaluate({"inlet.P": 3e5, "position": 0.7})

    For simple names without dots::

        map(MR=2.8, Pc=2e6)
    """

    def __init__(
        self,
        axes: List[str],
        outputs: List[str],
        _backend: Callable,
        _bounds: Optional[Dict[str, tuple]] = None,
        _extrapolation: str = "clamp",
        _source: str = "unknown",
        _axis_data: Optional[Dict[str, np.ndarray]] = None,
        _output_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.axes = list(axes)
        self.outputs = list(outputs)
        self._backend = _backend
        self._bounds: Dict[str, tuple] = _bounds or {}
        self._extrapolation = _extrapolation
        self._source = _source
        self._axis_data: Dict[str, np.ndarray] = _axis_data or {}
        self._output_data: Dict[str, np.ndarray] = _output_data or {}

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_arrays(
        cls,
        axes: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        extrapolation: str = "clamp",
    ) -> "PerformanceMap":
        """
        Structured (Cartesian product) grid.

        axes   : {name: 1-D array of coordinate values}
        outputs: {name: ndarray with shape = (len(a1), len(a2), ...)}

        For a 1-D map, output array shape is (N,).
        For a 2-D map over axes A×B, output array shape is (len(A), len(B)).
        """
        axis_names = list(axes.keys())
        output_names = list(outputs.keys())
        grids = [np.asarray(v, dtype=float) for v in axes.values()]
        arrays = {k: np.asarray(v, dtype=float) for k, v in outputs.items()}
        bounds = {n: (float(g.min()), float(g.max())) for n, g in zip(axis_names, grids)}

        if len(axis_names) == 1:
            x = grids[0]

            def backend(av: dict) -> dict:
                xv = np.asarray(av[axis_names[0]], dtype=float)
                return {k: np.interp(xv, x, arrays[k]) for k in output_names}

        else:
            from scipy.interpolate import RegularGridInterpolator

            interpolators = {
                k: RegularGridInterpolator(
                    tuple(grids),
                    arrays[k],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                for k in output_names
            }

            def backend(av: dict) -> dict:  # type: ignore[misc]
                vals = [np.atleast_1d(np.asarray(av[n], dtype=float)) for n in axis_names]
                is_scalar = all(v.ndim == 0 or v.shape == (1,) for v in vals)
                pts = np.column_stack(np.broadcast_arrays(*vals))
                result = {k: interpolators[k](pts) for k in output_names}
                if is_scalar:
                    return {k: float(v.flat[0]) for k, v in result.items()}
                return result

        return cls(
            axes=axis_names,
            outputs=output_names,
            _backend=backend,
            _bounds=bounds,
            _extrapolation=extrapolation,
            _source="arrays",
            _axis_data={n: g for n, g in zip(axis_names, grids)},
            _output_data=arrays,
        )

    @classmethod
    def from_scattered(
        cls,
        points: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        extrapolation: str = "clamp",
    ) -> "PerformanceMap":
        """
        Scattered (non-gridded) data points.
        All arrays must have the same length N.
        Uses thin-plate-spline RBF interpolation.
        """
        from scipy.interpolate import RBFInterpolator

        axis_names = list(points.keys())
        output_names = list(outputs.keys())
        pts = np.column_stack(
            [np.asarray(points[n], dtype=float) for n in axis_names]
        )
        out_arrays = {k: np.asarray(outputs[k], dtype=float) for k in output_names}
        bounds = {
            n: (float(pts[:, i].min()), float(pts[:, i].max()))
            for i, n in enumerate(axis_names)
        }

        interpolators = {
            k: RBFInterpolator(pts, out_arrays[k], kernel="thin_plate_spline")
            for k in output_names
        }

        def backend(av: dict) -> dict:
            vals = [np.atleast_1d(np.asarray(av[n], dtype=float)) for n in axis_names]
            query = np.column_stack(np.broadcast_arrays(*vals))
            result = {k: interpolators[k](query) for k in output_names}
            if query.shape[0] == 1:
                return {k: float(v[0]) for k, v in result.items()}
            return result

        return cls(
            axes=axis_names,
            outputs=output_names,
            _backend=backend,
            _bounds=bounds,
            _extrapolation=extrapolation,
            _source="scattered",
            _axis_data={n: pts[:, i] for i, n in enumerate(axis_names)},
            _output_data=out_arrays,
        )

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        axes: List[str],
        outputs: List[str],
        bounds: Optional[Dict[str, tuple]] = None,
        extrapolation: str = "clamp",
    ) -> "PerformanceMap":
        """
        Callable map.

        fn receives axis values as keyword arguments and must return a dict::

            fn(MR=2.8, Pc=2e6) -> {"eta_cstar": 0.975}

        Axis names with dots cannot be passed as kwargs — use a wrapper::

            fn=lambda **kw: {"Cd": 0.71 + 0.02 * kw["inlet.P"] / 3e5}
        """

        def backend(av: dict) -> dict:
            return fn(**av)

        return cls(
            axes=list(axes),
            outputs=list(outputs),
            _backend=backend,
            _bounds=bounds or {},
            _extrapolation=extrapolation,
            _source="callable",
        )

    @classmethod
    def constant(cls, **values: float) -> "PerformanceMap":
        """
        Constant map with no axes. Always returns the same values.

        Useful to keep the map interface when a fixed scalar is sufficient::

            PerformanceMap.constant(eta_cstar=0.975, Cd=0.72)
        """
        fvalues = {k: float(v) for k, v in values.items()}

        def backend(_av: dict = None) -> dict:  # type: ignore[assignment]
            return dict(fvalues)

        return cls(
            axes=[],
            outputs=list(fvalues.keys()),
            _backend=backend,
            _source="constant",
            _output_data={k: np.array([v]) for k, v in fvalues.items()},
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        axes: List[str],
        outputs: List[str],
        extrapolation: str = "clamp",
    ) -> "PerformanceMap":
        """
        Load from CSV. First row is the header with column names.

        Auto-detects structured grid vs. scattered data:
        - If unique-value product equals row count → structured grid
        - Otherwise → scattered (RBF interpolation)
        """
        data: Dict[str, List[float]] = {n: [] for n in axes + outputs}
        with open(path, newline="") as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                for n in axes + outputs:
                    data[n].append(float(row[n]))

        axes_arr = {n: np.asarray(data[n]) for n in axes}
        out_arr = {n: np.asarray(data[n]) for n in outputs}

        if len(axes) > 1:
            unique = [np.unique(axes_arr[n]) for n in axes]
            if np.prod([len(u) for u in unique]) == len(data[axes[0]]):
                shape = tuple(len(u) for u in unique)
                sorted_axes = {n: u for n, u in zip(axes, unique)}
                reshaped: Dict[str, np.ndarray] = {}
                rows_data = list(zip(*[data[n] for n in axes + outputs]))
                for out_name in outputs:
                    arr = np.empty(shape)
                    out_col = data[axes + outputs].index(out_name) if False else None
                    arr_flat = out_arr[out_name]
                    # Fill grid by matching each row to its grid index
                    row_axes = list(zip(*[data[n] for n in axes]))
                    for i, ax_vals in enumerate(row_axes):
                        idx = tuple(
                            int(np.searchsorted(unique[j], ax_vals[j]))
                            for j in range(len(axes))
                        )
                        arr[idx] = float(arr_flat[i])
                    reshaped[out_name] = arr
                return cls.from_arrays(sorted_axes, reshaped, extrapolation)

        return cls.from_scattered(axes_arr, out_arr, extrapolation)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, context: dict) -> dict:
        """
        Evaluate the map by extracting axis values from a context dict.

        context may contain any keys; only the declared axes are extracted.
        Extra keys are silently ignored, enabling direct pass-through of
        ``states | inputs`` dicts from component evaluation.

        Example::

            map.evaluate({"inlet.P": 3e5, "position": 0.7, "inlet.h": 1e5})
        """
        if not self.axes:
            return self._backend()

        missing = [n for n in self.axes if n not in context]
        if missing:
            raise KeyError(
                f"PerformanceMap axis {missing} not found in evaluation context.\n"
                f"Declared axes: {self.axes}\n"
                f"Available context keys: {sorted(context.keys())}"
            )

        axis_values = {n: context[n] for n in self.axes}
        axis_values = self._apply_bounds(axis_values)
        return self._backend(axis_values)

    def __call__(self, **kwargs) -> dict:
        """
        Convenience evaluation with keyword arguments.

        For axis names containing dots (e.g. ``"inlet.P"``), use
        :meth:`evaluate` instead::

            map(MR=2.8, Pc=2e6)                     # works
            map.evaluate({"inlet.P": 3e5})           # required for dotted names
        """
        return self.evaluate(kwargs)

    def _apply_bounds(self, av: dict) -> dict:
        if not self._bounds or self._extrapolation == "extrapolate":
            return av
        out = dict(av)
        for name, (lo, hi) in self._bounds.items():
            if name not in out:
                continue
            val = np.asarray(out[name], dtype=float)
            outside = np.any(val < lo) or np.any(val > hi)
            if outside:
                msg = (
                    f"PerformanceMap: axis '{name}' value {float(np.min(val)):.4g}–"
                    f"{float(np.max(val)):.4g} outside data bounds [{lo:.4g}, {hi:.4g}]"
                )
                if self._extrapolation == "error":
                    raise ValueError(msg)
                if self._extrapolation == "warn":
                    warnings.warn(msg, UserWarning, stacklevel=4)
            out[name] = np.clip(val, lo, hi)
        return out

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save to HDF5.

        Callable maps must be rasterized first via :meth:`rasterize`.
        """
        if self._source == "callable":
            raise ValueError(
                "Cannot save a callable PerformanceMap directly. "
                "Convert it first: map.rasterize({'axis': np.linspace(...)}).save(path)"
            )
        with h5py.File(path, "w") as f:
            f.attrs["source"] = self._source
            f.attrs["axes"] = json.dumps(self.axes)
            f.attrs["outputs"] = json.dumps(self.outputs)
            f.attrs["extrapolation"] = self._extrapolation

            ag = f.create_group("axes")
            for name, data in self._axis_data.items():
                ag.create_dataset(name, data=data)

            og = f.create_group("outputs")
            if self._source == "constant":
                vals = self._backend()
                for name in self.outputs:
                    og.attrs[name] = vals[name]
            else:
                for name, data in self._output_data.items():
                    og.create_dataset(name, data=data)

    @classmethod
    def load(cls, path: str) -> "PerformanceMap":
        """Load from HDF5."""
        with h5py.File(path, "r") as f:
            source = str(f.attrs["source"])
            axes = json.loads(str(f.attrs["axes"]))
            outputs = json.loads(str(f.attrs["outputs"]))
            extrapolation = str(f.attrs["extrapolation"])

            if source == "constant":
                values = {k: float(f["outputs"].attrs[k]) for k in outputs}
                return cls.constant(**values)

            axis_data = {n: f["axes"][n][()] for n in axes}
            output_data = {n: f["outputs"][n][()] for n in outputs}

        if source == "arrays":
            return cls.from_arrays(axis_data, output_data, extrapolation)
        if source == "scattered":
            return cls.from_scattered(axis_data, output_data, extrapolation)
        raise ValueError(f"Unknown map source '{source}' in {path}")

    def rasterize(self, grid: Dict[str, np.ndarray]) -> "PerformanceMap":
        """
        Convert a callable map to a structured array map by evaluating on
        a user-supplied grid. Required before saving a callable map.

        grid: {axis_name: 1-D array of values to evaluate at}
        """
        import itertools

        keys = list(grid.keys())
        missing = set(self.axes) - set(keys)
        if missing:
            raise ValueError(f"rasterize() grid missing axes: {missing}")

        arrays = [grid[k] for k in keys]
        out_arrays: Dict[str, np.ndarray] = {
            k: np.empty([len(a) for a in arrays]) for k in self.outputs
        }
        for idx in itertools.product(*[range(len(a)) for a in arrays]):
            ctx = {k: float(arrays[i][idx[i]]) for i, k in enumerate(keys)}
            result = self.evaluate(ctx)
            for k in self.outputs:
                out_arrays[k][idx] = result[k]

        return PerformanceMap.from_arrays(
            {k: grid[k] for k in keys}, out_arrays, self._extrapolation
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PerformanceMap(axes={self.axes}, outputs={self.outputs}, "
            f"source='{self._source}', extrapolation='{self._extrapolation}')"
        )
