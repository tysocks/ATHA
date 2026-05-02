# atha/validation/ttbe.py
"""
Simplified TTBE (Technology Test Bed Engine) model for validation.

Reference: ROCKETS System document (NASA/P&W 1991), Section 5.
Engine type: Staged combustion, LOX/LH2
This simplified model covers: combustion chamber + nozzle only.
Full turbomachinery not included in Phase 9 (requires pump maps).

Approximate TTBE parameters at 100% RPL:
- Main chamber pressure: ~20.6 MPa
- MR (O/F): ~6.0
- Throat area: 0.0687 m² (derived from 18 klbf engine class)
- Expansion ratio: 77.5
- Isp (vacuum): ~453 s
"""
from __future__ import annotations
from dataclasses import dataclass
from atha.thermo.ideal_gas import IdealGasBackend
from atha.jannaf.simplified import SimplifiedJANNAF
from atha.jannaf.efficiency import JANNAFEfficiencies
from atha.jannaf.performance import PerformanceResult


# LOX/LH2 combustion products at MR=6.0, Pc=20.6 MPa
# Mean MW ≈ 11.7 g/mol (H2O + H2 mixture at fuel-rich MR=6)
# R = R_universal / MW = 8314 / 11.7 = 711 J/(kg·K)
# gamma ≈ 1.24 (product mixture, slightly above 1.22 at this MR)
# These parameters reproduce c* ≈ 2365 m/s (matching ROCKETS reference)
LOX_LH2_GAMMA = 1.24
LOX_LH2_R = 711.0  # J/(kg·K)

# TTBE 100% RPL parameters (simplified)
TTBE_Pc = 20.6e6      # Pa, main chamber pressure
TTBE_MR = 6.0         # O/F mass ratio
TTBE_T_ad = 3560.0    # K, adiabatic flame temperature (approx)
TTBE_At = 0.0687      # m², throat area
TTBE_Ae_At = 77.5     # expansion ratio
TTBE_mdot = 468.0     # kg/s, total propellant flow


@dataclass
class TTBEResult:
    Isp_vacuum: float    # s
    thrust_vacuum: float # N
    c_star: float        # m/s
    Cf_vacuum: float
    MR: float
    Pc: float           # Pa
    performance: PerformanceResult


def run_ttbe_100pct_rpl() -> TTBEResult:
    """Run simplified TTBE at 100% RPL and return performance metrics."""
    thermo = IdealGasBackend(gamma=LOX_LH2_GAMMA, R=LOX_LH2_R)
    eff = JANNAFEfficiencies(
        eta_cstar=0.975,
        eta_Cd=0.98,
        eta_velocity=0.99,
        eta_divergence=0.9830,  # 15° half-angle nozzle
        eta_two_phase=1.0,
        eta_boundary_layer=0.99,
    )
    jannaf = SimplifiedJANNAF(
        thermo=thermo,
        efficiencies=eff,
        throat_area=TTBE_At,
        exit_area=TTBE_At * TTBE_Ae_At,
        ambient_pressure=0.0,  # vacuum
    )
    result = jannaf.compute(
        P_chamber=TTBE_Pc,
        T_chamber=TTBE_T_ad,
        MR=TTBE_MR,
        mdot_total=TTBE_mdot,
    )
    return TTBEResult(
        Isp_vacuum=result.Isp,
        thrust_vacuum=result.thrust,
        c_star=result.c_star,
        Cf_vacuum=result.Cf,
        MR=TTBE_MR,
        Pc=TTBE_Pc,
        performance=result,
    )
