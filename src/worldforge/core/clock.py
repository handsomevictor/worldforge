"""Abstract clock interface and DiscreteClock implementation."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Clock(ABC):
    """Abstract base class for all simulation clocks."""

    @abstractmethod
    def tick(self) -> None:
        """Advance the clock by one step."""

    @property
    @abstractmethod
    def now(self):
        """Return the current simulation time."""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Return True when the simulation should stop."""

    def reset(self) -> None:
        """Reset the clock to its initial state."""
        raise NotImplementedError(f"{type(self).__name__} does not support reset()")


class DiscreteClock(Clock):
    """Fixed-step integer clock. Runs for exactly `steps` ticks (step 0 to steps-1)."""

    def __init__(self, steps: int) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self._steps = steps
        self._current: int = 0

    def tick(self) -> None:
        self._current += 1

    @property
    def now(self) -> int:
        return self._current

    @property
    def is_done(self) -> bool:
        return self._current >= self._steps

    def reset(self) -> None:
        self._current = 0

    def __repr__(self) -> str:
        return f"DiscreteClock(steps={self._steps}, now={self._current})"
