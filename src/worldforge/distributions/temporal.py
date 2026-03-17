"""Temporal distributions: time-aware multipliers and seasonal wrappers."""
from __future__ import annotations

from typing import Optional

import numpy as np

from worldforge.distributions.base import Distribution


class HourOfDay:
    """
    Hour-of-day multiplier. Maps each hour (0-23) to a scalar multiplier.

    Hours not explicitly specified inherit from the nearest prior defined hour.
    If no prior hour exists, uses the smallest defined hour.
    """

    def __init__(self, pattern: dict[int, float]) -> None:
        if not pattern:
            raise ValueError("pattern must not be empty")
        self.pattern = pattern
        self._sorted_hours = sorted(pattern.keys())

    def get_multiplier(self, now=None) -> float:
        if now is None:
            return 1.0
        hour = now.hour if hasattr(now, "hour") else int(now) % 24
        # Walk backward from the current hour to find the most recent defined hour
        for h in reversed(self._sorted_hours):
            if h <= hour:
                return self.pattern[h]
        # Wrap around: use the last defined hour of the previous day
        return self.pattern[self._sorted_hours[-1]]

    def __repr__(self) -> str:
        return f"HourOfDay(hours={self._sorted_hours})"


class DayOfWeek:
    """
    Day-of-week multiplier. Keys are weekday names (Mon, Tue, Wed, Thu, Fri, Sat, Sun)
    or integers (0=Monday .. 6=Sunday).
    """

    _NAME_TO_INT = {
        "mon": 0, "monday": 0,
        "tue": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }

    def __init__(self, pattern: dict) -> None:
        if not pattern:
            raise ValueError("pattern must not be empty")
        # Normalize keys to integers 0-6
        self._pattern: dict[int, float] = {}
        for k, v in pattern.items():
            if isinstance(k, str):
                normalized = k.lower()
                if normalized not in self._NAME_TO_INT:
                    raise ValueError(f"Unknown day name: {k!r}")
                self._pattern[self._NAME_TO_INT[normalized]] = v
            else:
                self._pattern[int(k)] = v

    def get_multiplier(self, now=None) -> float:
        if now is None:
            return 1.0
        # datetime.weekday(): Monday=0, Sunday=6
        weekday = now.weekday() if hasattr(now, "weekday") else int(now) % 7
        return self._pattern.get(weekday, 1.0)

    def __repr__(self) -> str:
        return f"DayOfWeek(days={list(self._pattern.keys())})"


class Seasonal(Distribution):
    """
    Wraps a base distribution and scales its output by time-of-day / day-of-week
    multipliers.

    When used as a field default (no time context), multiplier=1.0 (passthrough).
    When called from step() or probes, pass `now=ctx.now` to apply temporal scaling.
    """

    def __init__(
        self,
        base: Distribution,
        hour_multiplier: Optional[HourOfDay] = None,
        day_multiplier: Optional[DayOfWeek] = None,
    ) -> None:
        self.base = base
        self.hour_multiplier = hour_multiplier
        self.day_multiplier = day_multiplier

    def _compute_multiplier(self, now=None) -> float:
        m = 1.0
        if now is not None and self.hour_multiplier is not None:
            m *= self.hour_multiplier.get_multiplier(now)
        if now is not None and self.day_multiplier is not None:
            m *= self.day_multiplier.get_multiplier(now)
        return m

    def sample(self, rng: np.random.Generator, now=None) -> float:
        return float(self.base.sample(rng)) * self._compute_multiplier(now)

    def sample_batch(self, n: int, rng: np.random.Generator, now=None) -> np.ndarray:
        return self.base.sample_batch(n, rng) * self._compute_multiplier(now)

    def mean(self) -> float:
        return self.base.mean()

    def std(self) -> float:
        return self.base.std()

    def __repr__(self) -> str:
        return f"Seasonal(base={self.base!r})"
