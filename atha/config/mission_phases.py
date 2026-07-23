"""Mission-phase control helpers for phase-aware engine execution.

ATHA phases are named time windows declared in ``analysis.time.phases``.
This module formalizes the control-side semantics that sit on top of those
windows:

- controller activation / deactivation by phase,
- optional integral / rate-limiter reset on phase entry,
- optional command hold when a controller is inactive,
- phase transition detection for the DAE loop,
- optional guard-based early advance (``advance_when``).
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


@dataclass(frozen=True)
class PhaseAdvanceGuard:
    """Threshold guard that can end a phase before its scheduled ``end_s``."""

    path: str
    op: str
    value: float


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
    """Resolve the active named phase at time ``t`` using scheduled windows only.

    Intervals are half-open ``[start_s, end_s)`` except the final sample at
    ``time_end_s``, which remains associated with the last phase that ends
    there. Guard-based early advances are applied by
    ``resolve_phase_name_with_guards``.
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


def parse_advance_guard(raw: Mapping[str, Any] | None) -> PhaseAdvanceGuard | None:
    """Parse an ``advance_when`` mapping into a typed guard."""

    if raw is None:
        return None
    path = raw.get("path", raw.get("measurement", raw.get("signal")))
    if not isinstance(path, str) or not path:
        raise ValueError("advance_when.path must be a non-empty string")
    op = str(raw.get("op", raw.get("operator", ">=")))
    if "value" not in raw:
        raise ValueError("advance_when.value is required")
    return PhaseAdvanceGuard(path=path, op=op, value=float(raw["value"]))


def evaluate_advance_guard(guard: PhaseAdvanceGuard, measurements: Mapping[str, Any]) -> bool:
    """Return True when the guard threshold is satisfied by ``measurements``."""

    value = _lookup_measurement(measurements, guard.path)
    if value is None:
        return False
    op = guard.op
    threshold = float(guard.value)
    if op in {">=", "ge"}:
        return value >= threshold
    if op in {">", "gt"}:
        return value > threshold
    if op in {"<=", "le"}:
        return value <= threshold
    if op in {"<", "lt"}:
        return value < threshold
    if op in {"==", "eq"}:
        return abs(value - threshold) <= 1.0e-12 * max(abs(threshold), 1.0)
    raise ValueError(f"unsupported advance_when.op {op!r}")


def resolve_phase_name_with_guards(
    phases: list[Any],
    t: float,
    *,
    time_end_s: float | None = None,
    measurements: Mapping[str, Any] | None = None,
    forced_end_times: Mapping[str, float] | None = None,
) -> str | None:
    """Resolve the active phase, honoring optional early-advance ends.

    ``forced_end_times`` maps phase name -> earliest effective end time once a
    guard has fired. When a phase ends early, the next phase start is pulled
    forward to that end so the sequencer does not leave a dead gap.

    Timed phases without guards behave exactly as ``resolve_phase_name``.
    ``measurements`` is accepted for API symmetry with
    ``update_forced_phase_ends`` but is not evaluated here — callers should
    refresh ``forced_end_times`` first.
    """

    _ = measurements
    t = float(t)
    forced = dict(forced_end_times or {})
    if not forced:
        return resolve_phase_name(phases, t, time_end_s)

    previous_ended_early = False
    cursor_end: float | None = None
    last_named: str | None = None
    last_scheduled_end: float | None = None

    for phase in phases:
        name = str(getattr(phase, "name", "") or "")
        if not name:
            continue
        scheduled_start = float(getattr(phase, "start_s"))
        scheduled_end = float(getattr(phase, "end_s"))
        start = scheduled_start
        if previous_ended_early and cursor_end is not None:
            start = min(scheduled_start, cursor_end)
        end = float(forced[name]) if name in forced else scheduled_end
        if end < start:
            end = start
        if start <= t < end:
            return name
        last_named = name
        last_scheduled_end = scheduled_end
        if name in forced:
            previous_ended_early = True
            cursor_end = end
        else:
            previous_ended_early = False
            cursor_end = scheduled_end

    if (
        time_end_s is not None
        and last_named is not None
        and last_scheduled_end is not None
        and t == float(time_end_s)
        and last_scheduled_end == float(time_end_s)
    ):
        return last_named
    return None


def update_forced_phase_ends(
    phases: list[Any],
    t: float,
    measurements: Mapping[str, Any],
    forced_end_times: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate ``advance_when`` guards and record early phase end times."""

    updated = dict(forced_end_times or {})
    t = float(t)
    active = resolve_phase_name_with_guards(
        phases,
        t,
        forced_end_times=updated,
    )
    for phase in phases:
        name = str(getattr(phase, "name", "") or "")
        if not name or name in updated:
            continue
        if active is not None and name != active:
            continue
        start = float(getattr(phase, "start_s"))
        end = float(getattr(phase, "end_s"))
        if not (start <= t < end) and name != active:
            continue
        raw_guard = getattr(phase, "advance_when", None)
        if raw_guard is None and isinstance(phase, Mapping):
            raw_guard = phase.get("advance_when")
        guard = parse_advance_guard(raw_guard if isinstance(raw_guard, Mapping) else None)
        if guard is None:
            continue
        if evaluate_advance_guard(guard, measurements):
            updated[name] = t
    return updated


def _lookup_measurement(measurements: Mapping[str, Any], path: str) -> float | None:
    key = path
    if key.startswith("measurements."):
        key = key[len("measurements.") :]
    candidates = (key, key.replace("_", "."), key.replace(".", "_"))
    for candidate in candidates:
        if candidate in measurements:
            value = measurements[candidate]
            return float(value) if isinstance(value, (int, float)) else None
    return None


def _phase_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(phase) for phase in value}
