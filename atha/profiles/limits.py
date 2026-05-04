# atha/profiles/limits.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


class EngineAbort(Exception):
    def __init__(self, reason: str, t: float = 0.0):
        super().__init__(f"EngineAbort at t={t:.4f}s: {reason}")
        self.reason = reason
        self.t = t


@dataclass
class SafetyLimit:
    name: str
    component_name: str
    state_name: str
    upper_limit: Optional[float] = None
    lower_limit: Optional[float] = None
    is_hard: bool = True
    description: str = ""


class AbortManager:
    def __init__(self, limits: List[SafetyLimit]):
        self.limits = limits

    def check(self, layout, X: np.ndarray, t: float = 0.0) -> None:
        """Check all limits against current state. Raises EngineAbort on hard violation."""
        for limit in self.limits:
            value = self._extract(layout, limit)
            if value is None:
                continue
            if limit.upper_limit is not None and value > limit.upper_limit:
                if limit.is_hard:
                    raise EngineAbort(
                        f"{limit.name}: {limit.component_name}.{limit.state_name} "
                        f"= {value:.4g} > upper limit {limit.upper_limit:.4g}", t=t
                    )
            if limit.lower_limit is not None and value < limit.lower_limit:
                if limit.is_hard:
                    raise EngineAbort(
                        f"{limit.name}: {limit.component_name}.{limit.state_name} "
                        f"= {value:.4g} < lower limit {limit.lower_limit:.4g}", t=t
                    )

    def _extract(self, layout, limit: SafetyLimit) -> Optional[float]:
        for comp in layout.components:
            if comp.name == limit.component_name:
                return comp._state_values.get(limit.state_name)
        return None

    def _extract_from_X(self, layout, limit: SafetyLimit, X: np.ndarray) -> Optional[float]:
        """Extract limit's state value directly from the state vector X."""
        for comp in layout.components:
            if comp.name == limit.component_name:
                off = layout.state_offsets.get(comp.name)
                if off is None:
                    return None
                for i, sname in enumerate(comp.state_names):
                    if sname == limit.state_name:
                        return float(X[off + i])
        return None

    def as_scipy_events(self, layout):
        """Return list of scipy event callables for solve_ivp."""
        events = []
        for limit in self.limits:
            if limit.upper_limit is not None:
                def upper_fn(t, X, _layout=layout, _limit=limit):
                    val = self._extract_from_X(_layout, _limit, X)
                    return (val - _limit.upper_limit) if val is not None else 1.0
                upper_fn.terminal = limit.is_hard
                upper_fn.direction = 1
                events.append(upper_fn)
            if limit.lower_limit is not None:
                def lower_fn(t, X, _layout=layout, _limit=limit):
                    val = self._extract_from_X(_layout, _limit, X)
                    return (_limit.lower_limit - val) if val is not None else 1.0
                lower_fn.terminal = limit.is_hard
                lower_fn.direction = 1
                events.append(lower_fn)
        return events
