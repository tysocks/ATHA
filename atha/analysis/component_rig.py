# atha/analysis/component_rig.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from atha.core.component import BaseComponent
from atha.core.port import PortDomain, PortDirection


def _rig_inputs(component: BaseComponent, bcs: Mapping[str, float]) -> Dict[str, float]:
    """Map short rig BC keys to the names ``compute_outputs`` expects."""
    inputs: Dict[str, float] = dict(bcs)
    if "omega" in inputs and "shaft.omega" not in inputs:
        inputs["shaft.omega"] = float(inputs["omega"])
    return inputs


@dataclass(frozen=True)
class SweepAxis:
    """One independent variable for :class:`ComponentSweep`."""

    name: str
    values: np.ndarray


class ComponentRig:
    """
    Evaluate a single component in isolation given boundary-condition dicts.

    BC keys use the short port-oriented names expected by the component
    (e.g. ``inlet.P``, ``omega``); ``omega`` is mapped to ``shaft.omega`` for
    pumps/turbines.
    """

    def __init__(self, component: BaseComponent) -> None:
        self.component = component

    def required_inputs(self) -> List[str]:
        """Nominal keys for a minimal pump-style rig (extend as needed)."""
        c = self.component
        if c.__class__.__name__ == "Pump":
            return ["inlet.P", "inlet.h", "inlet.mdot", "omega"]
        if c.__class__.__name__ == "Rotor":
            return ["omega"]
        keys: List[str] = []
        for pname in c.ports:
            port = c.ports[pname]
            if port.domain == PortDomain.FLUID:
                if port.direction == PortDirection.INLET:
                    keys.extend([f"{pname}.P", f"{pname}.h", f"{pname}.mdot"])
            elif port.domain == PortDomain.SHAFT:
                keys.append("omega")
        return sorted(set(keys)) or ["inlet.P", "inlet.h", "inlet.mdot", "omega"]

    def evaluate(self, bcs: Mapping[str, float]) -> Dict[str, Any]:
        comp = self.component
        inputs = _rig_inputs(comp, bcs)
        states = {k: comp._state_values[k] for k in comp.state_names}
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


class SweepResult:
    """Grid output from :meth:`ComponentSweep.run`."""

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

    def plot_map(
        self,
        z_key: str,
        *,
        x_axis: str,
        y_axis: str,
        x_label: str = "",
        y_label: str = "",
        y_scale: float = 1.0,
        y_scale_display: Optional[float] = None,
        colorbar_label: str = "",
        title: str = "",
        show: bool = True,
    ):
        import matplotlib.pyplot as plt

        if z_key not in self.outputs:
            raise KeyError(f"No output '{z_key}' in sweep (have {list(self.outputs)})")
        xv = self.axes[x_axis]
        yv = self.axes[y_axis] * y_scale
        Z = self.outputs[z_key].reshape(len(self.axes[y_axis]), len(self.axes[x_axis]))
        fig, ax = plt.subplots(figsize=(8, 5))
        X, Y = np.meshgrid(xv, yv)
        pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
        ax.set_xlabel(x_label or x_axis)
        disp = y_scale_display if y_scale_display is not None else y_scale
        if disp != 1.0:
            ax.set_ylabel(y_label + f" (×{disp:g})" if y_label else y_axis)
        else:
            ax.set_ylabel(y_label or y_axis)
        ax.set_title(title)
        plt.colorbar(pcm, ax=ax, label=colorbar_label or z_key)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def save(self, filename: str) -> None:
        import h5py

        with h5py.File(filename, "w") as f:
            for k, v in self.axes.items():
                f.create_dataset(f"axis/{k}", data=v)
            for k, v in self.outputs.items():
                f.create_dataset(f"output/{k}", data=v)
            f.create_dataset("failed", data=self.failed_mask.astype(np.uint8))


class ComponentSweep:
    """
    Cartesian product sweep over one or more :class:`SweepAxis` entries.
    """

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
            raise NotImplementedError("ComponentSweep supports 1–2 axes only for now")

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

        shape = (len(grids[1]), len(grids[0])) if len(grids) == 2 else (len(grids[0]),)
        if len(grids) == 2:
            for k in out_dict:
                out_dict[k] = out_dict[k].reshape(shape)

        axes_out = {self.axes[0].name: grids[0]}
        if len(grids) == 2:
            axes_out[self.axes[1].name] = grids[1]

        return SweepResult(axes_out, out_dict, failed.reshape(shape) if len(grids) == 2 else failed)
