from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import TextIO


@dataclass
class SolverProgressEvent:
    kind: str
    message: str
    time_s: float | None = None
    percent: float | None = None
    phase: str | None = None
    residual_name: str | None = None
    residual_value: float | None = None


class ConsoleProgressReporter:
    """Small terminal progress reporter for long ATHA solver runs."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        min_interval_s: float = 1.0,
        percent_step: float = 2.0,
    ) -> None:
        self.stream = stream or sys.stderr
        self.min_interval_s = float(min_interval_s)
        self.percent_step = max(float(percent_step), 0.1)
        self._last_emit = 0.0
        self._last_percent = -1.0e9
        self._last_line_len = 0

    def __call__(self, event: SolverProgressEvent) -> None:
        now = time.monotonic()
        if event.kind == "progress":
            percent = 0.0 if event.percent is None else float(event.percent)
            if (
                percent < 100.0
                and percent - self._last_percent < self.percent_step
                and now - self._last_emit < self.min_interval_s
            ):
                return
            self._last_percent = percent
            self._last_emit = now
            self._write(self._format_progress(event), carriage=True)
            return
        self._last_emit = now
        self._write(self._format_event(event), carriage=False)

    def finish(self) -> None:
        if self._last_line_len:
            self.stream.write("\n")
            self.stream.flush()
            self._last_line_len = 0

    def _format_progress(self, event: SolverProgressEvent) -> str:
        parts = ["ATHA"]
        if event.percent is not None:
            parts.append(f"{event.percent:6.2f}%")
        if event.time_s is not None:
            parts.append(f"t={event.time_s:.3f}s")
        if event.phase:
            parts.append(f"phase={event.phase}")
        if event.residual_name:
            parts.append(f"maxR={event.residual_name}:{event.residual_value:.3e}")
        parts.append(event.message)
        return " | ".join(parts)

    def _format_event(self, event: SolverProgressEvent) -> str:
        prefix = "ATHA"
        if event.kind:
            prefix = f"{prefix} {event.kind}"
        return f"{prefix}: {event.message}"

    def _write(self, text: str, *, carriage: bool) -> None:
        if self._last_line_len and not carriage:
            self.stream.write("\n")
            self._last_line_len = 0
        if carriage:
            padded = text.ljust(self._last_line_len)
            self.stream.write("\r" + padded)
            self._last_line_len = len(text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()


def should_enable_console_progress(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    return bool(getattr(sys.stderr, "isatty", lambda: False)())
