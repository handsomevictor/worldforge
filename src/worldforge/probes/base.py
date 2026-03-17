"""Abstract base class and interval utilities for all probes."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any


def _resolve_every(every: Any, clock: Any) -> int:
    """
    Convert the 'every' parameter to a step count.

    - int   → use directly
    - str   → parse '1 day', '1 week', etc.; convert to steps via the clock step
    - other → default 1
    """
    if isinstance(every, int):
        return max(1, every)

    if isinstance(every, str):
        from worldforge.time.calendar import CalendarClock, parse_duration
        if isinstance(clock, CalendarClock):
            interval_td = parse_duration(every)
            steps = max(1, round(interval_td / clock.step))
            return steps
        # DiscreteClock: accept "N <unit>" and just use N
        try:
            return max(1, int(every.strip().split()[0]))
        except (ValueError, IndexError):
            return 1

    return 1


class Probe:
    """
    Abstract base for all worldforge data-collection probes.

    Subclasses implement `collect(ctx)` which is called at configured intervals,
    and `finalize()` which returns the accumulated records as a list of dicts.
    """

    name: str = ""  # override in subclasses or set at construction

    def __init__(self, every: Any = 1, name: str = "") -> None:
        self.every = every
        if name:
            self.name = name
        self._step_interval: int = 1  # resolved by configure()

    def configure(self, clock: Any) -> None:
        """Called once before the simulation starts to resolve the interval."""
        self._step_interval = _resolve_every(self.every, clock)
        if not self.name:
            self.name = type(self).__name__.lower().replace("probe", "")

    def on_step(self, ctx: Any, step: int) -> None:
        """Called after each tick. Delegates to collect() when interval is hit."""
        if step % self._step_interval == 0:
            self.collect(ctx)

    @abstractmethod
    def collect(self, ctx: Any) -> None:
        """Collect data for the current tick."""

    @abstractmethod
    def finalize(self) -> list:
        """Return all collected records as a list of dicts."""
