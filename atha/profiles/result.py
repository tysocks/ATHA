# atha/profiles/result.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class PhaseResult:
    name: str
    t: np.ndarray           # shape (N,), time relative to phase start
    X: np.ndarray           # shape (N, n_states)
    state_names: List[str]  # ["comp.state", ...]
    X_final: np.ndarray     # shape (n_states,)
    abort_triggered: bool

    def get(self, component_name: str, state_name: str) -> np.ndarray:
        key = f"{component_name}.{state_name}"
        for i, n in enumerate(self.state_names):
            if n == key:
                return self.X[:, i]
        raise KeyError(f"State '{key}' not found. Available: {self.state_names}")

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 0 else 0.0


@dataclass
class TestProfileResult:
    profile_name: str
    phases: List[PhaseResult]
    state_names: List[str]
    abort_reason: Optional[str] = None
    abort_time: Optional[float] = None

    @property
    def success(self) -> bool:
        return self.abort_reason is None

    @property
    def total_duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def get_phase(self, name: str) -> PhaseResult:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(f"Phase '{name}' not found")

    def get_combined(self, component_name: str, state_name: str):
        """Concatenate a state across all phases with global time offset."""
        t_parts, X_parts = [], []
        t_offset = 0.0
        for phase in self.phases:
            try:
                X_part = phase.get(component_name, state_name)
                t_parts.append(phase.t + t_offset)
                X_parts.append(X_part)
            except KeyError:
                pass
            t_offset += phase.duration
        if not t_parts:
            raise KeyError(f"{component_name}.{state_name} not found in any phase")
        return np.concatenate(t_parts), np.concatenate(X_parts)

    def plot_timeline(self, states=None, show=True):
        """Plot selected state time-series with phase boundaries marked."""
        import matplotlib.pyplot as plt

        if states is None and self.state_names:
            states = self.state_names[:4]

        fig, axes = plt.subplots(len(states), 1,
                                  figsize=(12, 3 * len(states)), sharex=True)
        if len(states) == 1:
            axes = [axes]

        t_offset = 0.0
        phase_boundaries = [0.0]
        phase_names = []
        for phase in self.phases:
            t_offset += phase.duration
            phase_boundaries.append(t_offset)
            phase_names.append(phase.name)

        for ax, state_key in zip(axes, states):
            comp, sname = state_key.split(".", 1)
            try:
                t_global, X_vals = self.get_combined(comp, sname)
                ax.plot(t_global, X_vals)
                ax.set_ylabel(state_key)
                ax.grid(True, alpha=0.3)
                for tb in phase_boundaries[1:-1]:
                    ax.axvline(tb, color="gray", linestyle="--", alpha=0.5)
            except KeyError:
                ax.set_ylabel(f"{state_key} (N/A)")

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Test Profile: {self.profile_name}", y=1.01)

        for i, (t_start, t_end, name) in enumerate(
            zip(phase_boundaries[:-1], phase_boundaries[1:], phase_names)
        ):
            axes[0].text(
                (t_start + t_end) / 2, axes[0].get_ylim()[1],
                name, ha="center", va="bottom", fontsize=8, rotation=0,
            )

        plt.tight_layout()
        if show:
            plt.show()
        return fig
