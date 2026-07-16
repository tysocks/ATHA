"""Mission-phase control helpers for phase-aware engine execution.

ATHA phases are named time windows declared in ``analysis.time.phases``.
This module formalizes the control-side semantics that sit on top of those
windows:

- controller activation / deactivation by phase,
- optional integral / rate-limiter reset on phase entry,
- optional command hold when a controller is inactive,
- phase transition detection for the DAE loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PhaseTransition:
    """Describes a transition between mission phases at a sample time."""

    time_s: float
    previous: str | None
    current: str | None

    @property
    def entered(self) -> bool:
        return self.current is not None and self.current != self.previous

    @property
    def exited(self) -> bool:
        return self.previous is not None and self.previous != self.current


def controller_is_active(controller: Mapping[str, Any], current_phase: str | None) -> bool:
    """Return whether a controller should evaluate in the current phase.

    Rules:
    - ``active_phases`` if present: controller is active only in those phases.
    - ``inactive_phases`` if present: controller is inactive in those phases.
    - If both are present, ``active_phases`` is the inclusion set and
      ``inactive_phases`` is applied as an exclusion filter on top.
    - If neither is present, the controller is always active.
    """

    active_phases = controller.get("active_phases")
    inactive_phases = controller.get("inactive_phases")
    if active_phases is None and inactive_phases is None:
        return True
    if active_phases is not None:
        active = _phase_set(active_phases)
        if current_phase not in active:
            return False
    if inactive_phases is not None:
        inactive = _phase_set(inactive_phases)
        if current_phase in inactive:
            return False
    return True


def controller_should_reset_on_enter(controller: Mapping[str, Any], phase: str | None) -> bool:
    """Return whether controller memory should reset when entering ``phase``."""

    if phase is None:
        return False
    reset_on_enter = controller.get("reset_on_enter", False)
    if reset_on_enter is True:
        return True
    if reset_on_enter is False or reset_on_enter is None:
        return False
    return phase in _phase_set(reset_on_enter)


def controller_hold_when_inactive(controller: Mapping[str, Any]) -> bool:
    """Return whether inactive controllers should freeze their last command."""

    return bool(controller.get("hold_when_inactive", True))


def detect_phase_transition(
    previous_phase: str | None,
    current_phase: str | None,
    time_s: float,
) -> PhaseTransition | None:
    """Return a transition object when the active phase name changes."""

    if previous_phase == current_phase:
        return None
    return PhaseTransition(time_s=float(time_s), previous=previous_phase, current=current_phase)


def resolve_phase_name(phases: list[Any], t: float, time_end_s: float | None = None) -> str | None:
    """Resolve the active named phase at time ``t``.

    Intervals are half-open ``[start_s, end_s)`` except the final sample at
    ``time_end_s``, which remains associated with the last phase that ends
    there.
    """

    t = float(t)
    for phase in phases:
        name = getattr(phase, "name", "") or ""
        if not name:
            continue
        start = float(getattr(phase, "start_s"))
        end = float(getattr(phase, "end_s"))
        if start <= t < end:
            return str(name)
        if time_end_s is not None and t == float(time_end_s) and end == float(time_end_s):
            return str(name)
    return None


def _phase_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(phase) for phase in value}
