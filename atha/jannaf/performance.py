# atha/jannaf/performance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from atha.jannaf.efficiency import JANNAFEfficiencies


@dataclass
class PerformanceResult:
    """
    Output of a JANNAF performance calculation.
    All quantities in SI units.
    """
    # Delivered (real) performance
    Isp:           float   # s, delivered specific impulse
    c_star:        float   # m/s, delivered characteristic velocity
    Cf:            float   # dimensionless, delivered thrust coefficient
    thrust:        float   # N, delivered thrust

    # Mass flow
    mdot_total:    float   # kg/s, total propellant mass flow

    # Ideal (reference) performance
    c_star_ideal:  float   # m/s
    Isp_ideal:     float   # s

    # Operating conditions
    MR:            float   # oxidizer/fuel mass ratio
    P_chamber:     float   # Pa
    T_chamber:     float   # K, adiabatic flame temperature
    expansion_ratio: float  # Ae/At

    # Nozzle exit
    P_exit:        float   # Pa
    V_exit:        float   # m/s, exit velocity

    # Efficiencies used
    efficiencies:  JANNAFEfficiencies

    # Uncertainty (optional, populated by uncertainty analysis)
    Isp_1sigma:    Optional[float] = None  # s
