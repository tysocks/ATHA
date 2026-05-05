# atha/analysis/component_rig.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from atha.core.component import BaseComponent
from atha.core.port import PortDomain, PortDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rig_inputs(component: BaseComponent, bcs: Mapping[str, float]) -> Dict[str, float]:
    """Normalise short rig BC keys → names compute_outputs expects."""
    inputs: Dict[str, float] = dict(bcs)
    if "omega" in inputs and "shaft.omega" not in inputs:
        inputs["shaft.omega"] = float(inputs["omega"])
    return inputs


def _rig_bcs_fn(comp_name: str, bcs_fn):
    """Wrap a bare-key BCS callable to add the component-name prefix for solvers."""
    def wrapped(t):
        raw = bcs_fn(t)
        result: Dict[str, float] = {}
        for k, v in raw.items():
            result[k] = v
            result[f"{comp_name}.{k}"] = v
        return result
    return wrapped


# ---------------------------------------------------------------------------
# RigResult
# ---------------------------------------------------------------------------

@dataclass
class RigResult:
    """Time-series result from ComponentRig.transient()."""

    component_name: str
    t: np.ndarray                        # shape (N,)
    outputs: Dict[str, np.ndarray]       # key → shape (N,)
    state_names: List[str]               # empty for algebraic components
    X: Optional[np.ndarray]             # shape (N, n_states), None for algebraic

    def get(self, key: str) -> np.ndarray:
        if key not in self.outputs:
            raise KeyError(
                f"Key '{key}' not in RigResult. Available: {sorted(self.outputs)}"
            )
        return self.outputs[key]

    def plot(self, *keys: str, title: str = "", xlabel: str = "Time [s]",
             show: bool = True):
        import matplotlib.pyplot as plt
        ks = list(keys) if keys else list(self.outputs)
        fig, axes = plt.subplots(len(ks), 1, figsize=(9, 3 * len(ks)), squeeze=False)
        for ax, k in zip(axes[:, 0], ks):
            ax.plot(self.t, self.outputs[k])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(k)
            ax.set_title(k)
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def save(self, path: str) -> None:
        import h5py
        with h5py.File(path, "w") as f:
            f.attrs["component_name"] = self.component_name
            f.create_dataset("t", data=self.t)
            for k, v in self.outputs.items():
                f.create_dataset(f"outputs/{k}", data=v)
            if self.X is not None:
                f.create_dataset("X", data=self.X)
            f.attrs["state_names"] = json_safe(self.state_names)

    @classmethod
    def load(cls, path: str) -> "RigResult":
        import h5py
        with h5py.File(path, "r") as f:
            comp_name = str(f.attrs["component_name"])
            t = f["t"][:]
            outputs = {k: f[f"outputs/{k}"][:] for k in f["outputs"]}
            X = f["X"][:] if "X" in f else None
            state_names = list(f.attrs.get("state_names", []))
        return cls(component_name=comp_name, t=t, outputs=outputs,
                   state_names=state_names, X=X)


def json_safe(lst):
    return [str(x) for x in lst]


# ---------------------------------------------------------------------------
# ComponentRig
# ---------------------------------------------------------------------------

class ComponentRig:
    """
    Evaluate a single component in isolation.

    For algebraic components (n_states == 0) evaluate() calls compute_outputs
    directly.  For dynamic components (n_states > 0) evaluate() runs
    SteadyStateSolver to find the steady-state operating point.

    BC keys use the short port-oriented names expected by the component
    (e.g. ``"inlet.P"``, ``"omega"``); ``"omega"`` is mapped to
    ``"shaft.omega"`` automatically.
    """

    def __init__(self, component: BaseComponent) -> None:
        self.component = component

    # ------------------------------------------------------------------
    # required_inputs
    # ------------------------------------------------------------------

    def required_inputs(self) -> List[str]:
        """Return the likely required BCS keys for this component."""
        c = self.component
        if c.__class__.__name__ == "Pump":
            return ["inlet.P", "inlet.h", "inlet.mdot", "omega"]
        if c.__class__.__name__ == "Rotor":
            return ["omega"]
        keys: List[str] = []
        for pname, port in c.ports.items():
            if port.domain == PortDomain.FLUID:
                if port.direction == PortDirection.INLET:
                    keys.extend([f"{pname}.P", f"{pname}.h", f"{pname}.mdot"])
            elif port.domain == PortDomain.SHAFT:
                keys.append("omega")
        return sorted(set(keys)) or ["inlet.P", "inlet.h", "inlet.mdot", "omega"]

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, bcs: Mapping[str, float]) -> Dict[str, Any]:
        """
        Single-point evaluation.

        Algebraic components (n_states == 0): calls compute_outputs directly.
        Dynamic components (n_states > 0): runs SteadyStateSolver to find the
        steady-state where all dX/dt = 0, then evaluates outputs at that point.
        Updates component._state_values in-place.
        """
        comp = self.component
        inputs = _rig_inputs(comp, bcs)

        if comp.n_states == 0:
            states = {}
            out = comp.compute_outputs(0.0, states, inputs)
            comp.last_outputs = out
        else:
            from atha.core.engine import Engine
            from atha.solver.steady_state import SteadyStateSolver

            eng = Engine("_rig")
            eng.add_component(comp)
            layout = eng.compile()
            X0 = layout.assemble_state_vector()

            # Prefix BCS keys with comp name so _component_inputs can find them
            bcs_prefixed = {}
            for k, v in bcs.items():
                bcs_prefixed[k] = v
                bcs_prefixed[f"{comp.name}.{k}"] = v

            X_ss = SteadyStateSolver(layout).solve(X0, bcs_prefixed)
            layout.scatter_state_vector(X_ss)
            states = {sname: comp._state_values[sname] for sname in comp.state_names}
            out = comp.compute_outputs(0.0, states, inputs)
            comp.last_outputs = out

        mapped = dict(out)
        if "efficiency" in mapped and "eta" not in mapped:
            mapped["eta"] = mapped["efficiency"]
        if "tau_load" in mapped and "tau" not in mapped:
            mapped["tau"] = mapped["tau_load"]
        if "tau_drive" in mapped and "tau" not in mapped:
            mapped["tau"] = mapped["tau_drive"]
        return mapped

    # ------------------------------------------------------------------
    # transient
    # ------------------------------------------------------------------

    def transient(
        self,
        t_span: Tuple[float, float],
        bcs_fn,
        X0: Optional[np.ndarray] = None,
        recording_rate_hz: float = 100.0,
    ) -> RigResult:
        """
        Time-domain simulation with time-varying boundary conditions.

        Algebraic components: evaluates compute_outputs at each sample of a
        uniform time grid (no ODE integration).

        Dynamic components: runs TransientSolver (Radau) and re-evaluates
        outputs at dense output points.
        """
        comp = self.component
        t0, t1 = float(t_span[0]), float(t_span[1])
        N = max(2, int(round((t1 - t0) * recording_rate_hz)) + 1)
        t_arr = np.linspace(t0, t1, N)

        if comp.n_states == 0:
            # Point-by-point evaluation
            out_list: List[Dict] = []
            for t in t_arr:
                raw = bcs_fn(t)
                inputs = _rig_inputs(comp, raw)
                states: Dict = {}
                out = comp.compute_outputs(t, states, inputs)
                out_list.append(out)

            outputs_arr = _stack_outputs(out_list)
            return RigResult(
                component_name=comp.name,
                t=t_arr,
                outputs=outputs_arr,
                state_names=[],
                X=None,
            )

        else:
            from atha.core.engine import Engine
            from atha.solver.transient import TransientSolver

            eng = Engine("_rig")
            eng.add_component(comp)
            layout = eng.compile()

            if X0 is None:
                X0 = layout.assemble_state_vector()

            wrapped_bcs = _rig_bcs_fn(comp.name, bcs_fn)
            dt = (t1 - t0) / (N - 1)
            sol = TransientSolver(layout, max_step=dt).integrate(
                (t0, t1), X0, wrapped_bcs
            )

            # Dense-output re-evaluation at uniform grid
            X_dense = sol.X  # already sampled at sol.t; re-sample to t_arr
            # sol.t may not match t_arr exactly; interpolate each state
            X_interp = np.column_stack([
                np.interp(t_arr, sol.t, sol.X[:, i])
                for i in range(sol.X.shape[1])
            ])

            out_list = []
            for j, t in enumerate(t_arr):
                layout.scatter_state_vector(X_interp[j])
                states = {sname: float(X_interp[j, i])
                          for i, sname in enumerate(comp.state_names)}
                raw = bcs_fn(t)
                inputs = _rig_inputs(comp, raw)
                out = comp.compute_outputs(t, states, inputs)
                out_list.append(out)

            outputs_arr = _stack_outputs(out_list)
            return RigResult(
                component_name=comp.name,
                t=t_arr,
                outputs=outputs_arr,
                state_names=comp.state_names,
                X=X_interp,
            )


def _stack_outputs(out_list: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert list of output dicts to dict of numpy arrays."""
    if not out_list:
        return {}
    keys = [k for k, v in out_list[0].items() if isinstance(v, (int, float))]
    result: Dict[str, np.ndarray] = {}
    for k in keys:
        try:
            result[k] = np.array([float(o.get(k, float("nan"))) for o in out_list])
        except (TypeError, ValueError):
            pass
    return result


# ---------------------------------------------------------------------------
# SweepAxis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepAxis:
    """One independent variable for ComponentSweep."""
    name: str
    values: np.ndarray


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------

class SweepResult:
    """Grid output from ComponentSweep.run()."""

    def __init__(
        self,
        axes: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        failed_mask: np.ndarray,
    ) -> None:
        self.axes = axes
        self.outputs = outputs
        self.failed_mask = failed_mask

    @property
    def n_failed(self) -> int:
        return int(np.sum(self.failed_mask))

    def get(self, key: str) -> np.ndarray:
        if key not in self.outputs:
            raise KeyError(f"No output '{key}' (have {list(self.outputs)})")
        return self.outputs[key]

    # ------------------------------------------------------------------
    # plot_map  (2-D contour / pcolormesh)
    # ------------------------------------------------------------------

    def plot_map(
        self,
        z_key: str,
        *,
        x_axis: str,
        y_axis: str,
        x_label: str = "",
        y_label: str = "",
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        y_scale_display: Optional[float] = None,
        colorbar_label: str = "",
        title: str = "",
        show: bool = True,
    ):
        import matplotlib.pyplot as plt

        if z_key not in self.outputs:
            raise KeyError(f"No output '{z_key}' (have {list(self.outputs)})")
        if len(self.axes) < 2:
            raise ValueError("plot_map requires a 2-axis sweep.")

        xv = self.axes[x_axis] * x_scale
        yv = self.axes[y_axis] * y_scale
        Z = self.outputs[z_key].reshape(len(self.axes[y_axis]), len(self.axes[x_axis]))
        fig, ax = plt.subplots(figsize=(8, 5))
        X, Y = np.meshgrid(xv, yv)
        pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
        ax.set_xlabel(x_label or x_axis)
        disp = y_scale_display if y_scale_display is not None else y_scale
        if disp != 1.0:
            ax.set_ylabel(f"{y_label or y_axis} (×{disp:g})")
        else:
            ax.set_ylabel(y_label or y_axis)
        ax.set_title(title)
        plt.colorbar(pcm, ax=ax, label=colorbar_label or z_key)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # plot_curve  (1-D line plot; works for 1- or 2-axis sweeps)
    # ------------------------------------------------------------------

    def plot_curve(
        self,
        output_key: str,
        sweep_axis: str,
        fixed: Optional[Dict[str, float]] = None,
        label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        ax=None,
        show: bool = True,
    ):
        """
        Plot a 1-D curve of *output_key* vs *sweep_axis*.

        For a 2-axis sweep, *fixed* pins the other axis to the nearest stored
        value.  For a 1-axis sweep, *fixed* is ignored.
        """
        import matplotlib.pyplot as plt

        if output_key not in self.outputs:
            raise KeyError(f"No output '{output_key}' (have {list(self.outputs)})")
        if sweep_axis not in self.axes:
            raise KeyError(f"Axis '{sweep_axis}' not in sweep axes {list(self.axes)}.")

        axis_names = list(self.axes)

        if len(axis_names) == 1:
            x_vals = self.axes[sweep_axis] * x_scale
            y_vals = self.outputs[output_key] * y_scale
        else:
            # 2-axis case: find the slice nearest the fixed value
            other_axes = [a for a in axis_names if a != sweep_axis]
            if fixed is None:
                fixed = {}
            # Determine which axis is which in the (row, col) storage
            # Storage convention: outputs shaped (len(axis[0]), len(axis[1]))
            sweep_idx = axis_names.index(sweep_axis)

            # For each fixed axis, find closest index
            slice_indices: List[slice | int] = [slice(None), slice(None)]
            for other in other_axes:
                other_idx = axis_names.index(other)
                target = fixed.get(other, self.axes[other][len(self.axes[other]) // 2])
                closest = int(np.argmin(np.abs(self.axes[other] - target)))
                slice_indices[other_idx] = closest

            z = self.outputs[output_key].reshape(
                len(self.axes[axis_names[0]]), len(self.axes[axis_names[1]])
            )
            y_vals = z[tuple(slice_indices)] * y_scale
            x_vals = self.axes[sweep_axis] * x_scale

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.figure

        ax.plot(x_vals, y_vals, label=label)
        ax.set_xlabel(xlabel or sweep_axis)
        ax.set_ylabel(ylabel or output_key)
        if label:
            ax.legend()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, filename: str) -> None:
        import h5py
        with h5py.File(filename, "w") as f:
            for k, v in self.axes.items():
                f.create_dataset(f"axis/{k}", data=v)
            for k, v in self.outputs.items():
                f.create_dataset(f"output/{k}", data=v)
            f.create_dataset("failed", data=self.failed_mask.astype(np.uint8))

    @classmethod
    def load(cls, filename: str) -> "SweepResult":
        import h5py
        with h5py.File(filename, "r") as f:
            axes = {k: f[f"axis/{k}"][:] for k in f["axis"]}
            outputs = {k: f[f"output/{k}"][:] for k in f["output"]}
            failed_mask = f["failed"][:].astype(bool)
        return cls(axes=axes, outputs=outputs, failed_mask=failed_mask)


# ---------------------------------------------------------------------------
# ComponentSweep
# ---------------------------------------------------------------------------

class ComponentSweep:
    """Cartesian product sweep over one or more SweepAxis entries."""

    def __init__(
        self,
        rig: ComponentRig,
        axes: Sequence[SweepAxis],
        fixed_bcs: Mapping[str, float],
        outputs: Sequence[str],
        n_jobs: int = 1,
    ) -> None:
        self.rig = rig
        self.axes = list(axes)
        self.fixed_bcs = dict(fixed_bcs)
        self.output_keys = list(outputs)
        self.n_jobs = n_jobs

    def run(self) -> SweepResult:
        axis_names = [a.name for a in self.axes]
        grids = [np.asarray(a.values, dtype=float) for a in self.axes]

        if len(grids) == 1:
            points = [(float(x),) for x in grids[0]]
        elif len(grids) == 2:
            g0, g1 = grids
            points = [
                (float(g0[i]), float(g1[j]))
                for i in range(len(g0))
                for j in range(len(g1))
            ]
        else:
            raise NotImplementedError("ComponentSweep supports 1–2 axes only.")

        def _one(p: Tuple[float, ...]) -> Tuple[Dict[str, float], bool]:
            bcs = dict(self.fixed_bcs)
            for name, val in zip(axis_names, p):
                bcs[name] = val
            try:
                ev = self.rig.evaluate(bcs)
                row = {k: float(ev[k]) for k in self.output_keys if k in ev}
                return row, False
            except Exception:
                return {k: float("nan") for k in self.output_keys}, True

        if self.n_jobs == 1:
            rows = [_one(p) for p in points]
        else:
            rows = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(_one)(p) for p in points
            )

        failed = np.array([r[1] for r in rows], dtype=bool)
        out_dict: Dict[str, np.ndarray] = {}
        for k in self.output_keys:
            out_dict[k] = np.array([r[0].get(k, float("nan")) for r in rows])

        if len(grids) == 2:
            shape = (len(grids[0]), len(grids[1]))
            for k in out_dict:
                out_dict[k] = out_dict[k].reshape(shape)
            failed = failed.reshape(shape)
            axes_out = {axis_names[0]: grids[0], axis_names[1]: grids[1]}
        else:
            axes_out = {axis_names[0]: grids[0]}

        return SweepResult(axes_out, out_dict, failed)
