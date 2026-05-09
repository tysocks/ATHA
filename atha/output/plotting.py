from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_telemetry(path: str | Path, telemetry_config, columns: Mapping[str, np.ndarray]) -> Path | None:
    """Create a generic telemetry plot from telemetry YAML plot definitions."""

    if telemetry_config is None or not telemetry_config.exports.get("plot", False):
        return None
    plot_specs = list(getattr(telemetry_config, "plots", []) or [])
    if not plot_specs:
        plot_specs = _default_plots(columns)
    if not plot_specs:
        return None

    out_path = Path(path)
    n = len(plot_specs)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, spec in zip(axes, plot_specs):
        x_name = str(spec.get("x", "TIME"))
        y_names = spec.get("y", [])
        if isinstance(y_names, str):
            y_names = [y_names]
        if x_name not in columns:
            raise ValueError(f"Telemetry plot x channel not found: {x_name}")
        for y_name in y_names:
            y_alias = str(y_name)
            if y_alias not in columns:
                raise ValueError(f"Telemetry plot y channel not found: {y_alias}")
            ax.plot(columns[x_name], columns[y_alias], label=y_alias)
        ax.set_title(str(spec.get("title", "")))
        ax.set_ylabel(str(spec.get("ylabel", "")))
        if len(y_names) > 1 or spec.get("legend", True):
            ax.legend()
    axes[-1].set_xlabel(str(plot_specs[-1].get("xlabel", "Time [s]")))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _default_plots(columns: Mapping[str, np.ndarray]) -> list[dict]:
    y = [name for name in columns if name != "TIME"]
    if not y:
        return []
    return [{"title": "Telemetry", "x": "TIME", "y": y, "ylabel": "Value"}]
