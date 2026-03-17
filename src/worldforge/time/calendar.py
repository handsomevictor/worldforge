"""CalendarClock: real wall-clock time with timezone support."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone as _tz
from typing import Union

from worldforge.core.clock import Clock
from worldforge.core.exceptions import ConfigurationError


def _parse_dt(dt: Union[str, datetime]) -> datetime:
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, str):
        return datetime.fromisoformat(dt)
    raise ConfigurationError(f"Cannot parse datetime: {dt!r}")


def parse_duration(dur: Union[str, timedelta]) -> timedelta:
    """Parse duration string like '1 day', '2 hours', '30 minutes'."""
    if isinstance(dur, timedelta):
        return dur
    if isinstance(dur, str):
        parts = dur.strip().split()
        if len(parts) < 2:
            raise ConfigurationError(f"Duration must be '<n> <unit>', got {dur!r}")
        n = int(parts[0])
        unit = parts[1].rstrip("s").lower()
        mapping = {
            "second":  timedelta(seconds=1),
            "minute":  timedelta(minutes=1),
            "hour":    timedelta(hours=1),
            "day":     timedelta(days=1),
            "week":    timedelta(weeks=1),
        }
        if unit not in mapping:
            raise ConfigurationError(
                f"Unknown time unit: {unit!r}. Valid: {list(mapping)}"
            )
        return mapping[unit] * n
    raise ConfigurationError(f"Cannot parse duration: {dur!r}")


class CalendarClock(Clock):
    """
    Real calendar-time clock.

    Advances by a fixed timedelta step on each tick.
    Simulation ends when `now >= end`.

    Parameters
    ----------
    start:    ISO date string or datetime.
    end:      ISO date string or datetime.
    step:     Duration string like '1 day', '1 hour', or a timedelta.
    timezone: Timezone label (informational; stored but not enforced).
    realtime: If True, sleep for the real-world equivalent of `step` between
              ticks. Useful for digital-twin / live-monitoring scenarios.

    Example::

        clock = CalendarClock(start="2024-01-01", end="2025-01-01", step="1 day")

        # Digital-twin: tick every real second (step = 1 minute compressed to 1 s)
        clock = CalendarClock(
            start="2024-01-01", end="2024-01-02",
            step="1 minute", realtime=True,
        )
    """

    def __init__(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        step: Union[str, timedelta],
        timezone: str = "UTC",
        realtime: bool = False,
        realtime_factor: float = 1.0,
    ) -> None:
        self._start = _parse_dt(start)
        self._end = _parse_dt(end)
        self._step = parse_duration(step)
        self._current = self._start
        self._timezone = timezone
        self._realtime = realtime
        self._realtime_factor = realtime_factor   # wall-seconds per sim-step

        if self._start >= self._end:
            raise ConfigurationError(
                f"start must be before end: {self._start} >= {self._end}"
            )

    def tick(self) -> None:
        if self._realtime:
            import time as _time
            _time.sleep(self._realtime_factor)
        self._current += self._step

    @property
    def now(self) -> datetime:
        return self._current

    @property
    def is_done(self) -> bool:
        return self._current >= self._end

    @property
    def step(self) -> timedelta:
        """The duration of each simulation step."""
        return self._step

    def reset(self) -> None:
        self._current = self._start

    def __repr__(self) -> str:
        return (
            f"CalendarClock(start={self._start.date()}, "
            f"end={self._end.date()}, step={self._step})"
        )
