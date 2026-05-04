# atha/profiles/phase.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List


class PhaseMode(Enum):
    STEADY_TRIM = "steady_trim"
    TRANSIENT = "transient"
    DWELL = "dwell"


@dataclass
class ControlCommand:
    bcs_key: str
    fn: Callable[[float], float]


@dataclass
class PhaseDefinition:
    name: str
    mode: PhaseMode
    duration: float
    trim_targets: Dict[str, float] = field(default_factory=dict)
    control_commands: List[ControlCommand] = field(default_factory=list)
    abort_checks: List = field(default_factory=list)
    recording_rate_hz: float = 100.0
    solver_options: Dict = field(default_factory=dict)
