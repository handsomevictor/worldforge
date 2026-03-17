"""EventDrivenClock: next-event time advance for sparse-event simulations."""
from __future__ import annotations

from worldforge.core.clock import Clock
from worldforge.core.exceptions import ConfigurationError, EventOrderError


class EventDrivenClock(Clock):
    """
    Clock that advances to the time of the next scheduled event.

    The simulation runner is responsible for calling `advance_to(t)` with
    the time of the next event from the EventQueue. The clock does not tick
    automatically; `tick()` is a no-op and advancement is via `advance_to()`.

    Example::

        clock = EventDrivenClock(max_time=1_000_000)
        # Runner: clock.advance_to(event_queue.peek_time())
    """

    def __init__(self, max_time: float = float("inf"), min_step: float = 1e-12) -> None:
        if max_time <= 0:
            raise ConfigurationError(f"max_time must be > 0, got {max_time}")
        self._max_time = max_time
        self._min_step = min_step
        self._current: float = 0.0

    def advance_to(self, t: float) -> None:
        """Advance clock to time `t`. Must be >= current time."""
        if t < self._current - self._min_step:
            raise EventOrderError(
                f"Cannot advance clock backward: {self._current} -> {t}"
            )
        self._current = float(t)

    def tick(self) -> None:
        """No-op for EventDrivenClock. Use advance_to() instead."""

    @property
    def now(self) -> float:
        return self._current

    @property
    def is_done(self) -> bool:
        return self._current >= self._max_time

    def reset(self) -> None:
        self._current = 0.0

    def __repr__(self) -> str:
        return f"EventDrivenClock(now={self._current}, max_time={self._max_time})"
