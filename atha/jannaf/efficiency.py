# atha/jannaf/efficiency.py
from __future__ import annotations
import math
from dataclasses import dataclass

_ETA_DIV_DEFAULT = 0.5 * (1.0 + math.cos(math.radians(15.0)))  # 15° half-angle conical


@dataclass
class JANNAFEfficiencies:
    """
    JANNAF efficiency factors for the simplified performance procedure.
    All factors dimensionless, in (0, 1].
    Reference: JANNAF CPIA-246, Section 3.
    """
    eta_cstar:          float = 0.975
    eta_Cd:             float = 0.98
    eta_velocity:       float = 0.99
    eta_divergence:     float = _ETA_DIV_DEFAULT   # 15° conical: 0.9830
    eta_two_phase:      float = 1.00
    eta_boundary_layer: float = 0.99

    def __post_init__(self) -> None:
        for fname, value in [
            ("eta_cstar",          self.eta_cstar),
            ("eta_Cd",             self.eta_Cd),
            ("eta_velocity",       self.eta_velocity),
            ("eta_divergence",     self.eta_divergence),
            ("eta_two_phase",      self.eta_two_phase),
            ("eta_boundary_layer", self.eta_boundary_layer),
        ]:
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    f"Efficiency factor {fname}={value:.4f} must be in (0, 1]."
                )

    @property
    def overall_Cf_efficiency(self) -> float:
        return self.eta_velocity * self.eta_divergence * self.eta_boundary_layer
