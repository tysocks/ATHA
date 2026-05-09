from __future__ import annotations

import numpy as np


def telemetry_times(telemetry_config, t0: float, tf: float, default_rate_hz: float = 100.0) -> np.ndarray:
    """Return uniformly spaced telemetry sample times including the final time."""

    rate = float(getattr(telemetry_config, "sample_rate_hz", None) or default_rate_hz)
    if rate <= 0.0:
        raise ValueError("telemetry sample_rate_hz must be positive")
    dt = 1.0 / rate
    t = np.arange(float(t0), float(tf) + 0.5 * dt, dt)
    t[-1] = min(t[-1], float(tf))
    if t[-1] < float(tf):
        t = np.append(t, float(tf))
    return t
