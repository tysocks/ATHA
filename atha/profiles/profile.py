# atha/profiles/profile.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np

from atha.profiles.phase import PhaseDefinition
from atha.profiles.limits import SafetyLimit, EngineAbort
from atha.profiles.executor import execute_phase
from atha.profiles.result import TestProfileResult


@dataclass
class TestProfile:
    name: str
    phases: List[PhaseDefinition]
    global_limits: List[SafetyLimit] = field(default_factory=list)

    def execute(
        self,
        layout,
        X0: np.ndarray,
        bcs_fn: Optional[Callable[[float], Dict[str, float]]] = None,
    ) -> TestProfileResult:
        """Run all phases sequentially; state threads from phase to phase.

        Optional ``bcs_fn(t)`` returns extra boundary-condition keys merged into
        transient phases (after control commands). Ignored for STEADY_TRIM/DWELL.
        """
        state_names = layout.all_state_names()
        phase_results = []
        X_current = X0.copy()

        for phase in self.phases:
            try:
                result = execute_phase(
                    layout, X_current, phase,
                    global_limits=self.global_limits,
                    extra_bcs_fn=bcs_fn,
                )
                phase_results.append(result)
                X_current = result.X_final.copy()

            except EngineAbort as e:
                return TestProfileResult(
                    profile_name=self.name,
                    phases=phase_results,
                    state_names=state_names,
                    abort_reason=e.reason,
                    abort_time=e.t,
                )

        return TestProfileResult(
            profile_name=self.name,
            phases=phase_results,
            state_names=state_names,
        )
