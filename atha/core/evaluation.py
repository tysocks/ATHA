from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EvaluationResult:
    dXdt: np.ndarray
    Rz: np.ndarray
    outputs: Dict[str, float]
    residual_names: List[str]
    output_names: List[str]
    residual_scales: Optional[np.ndarray] = None

    @property
    def normalized_residuals(self) -> np.ndarray:
        if self.residual_scales is None:
            return self.Rz
        scales = np.asarray(self.residual_scales, dtype=float)
        return self.Rz / scales

    def max_normalized_residual(self) -> Tuple[str, float]:
        if len(self.Rz) == 0:
            return "", 0.0
        normalized = self.normalized_residuals
        idx = int(np.argmax(np.abs(normalized)))
        return self.residual_names[idx], float(normalized[idx])
