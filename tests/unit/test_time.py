"""Unit tests for CalendarClock and EventDrivenClock."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from worldforge.core.exceptions import ConfigurationError, EventOrderError
from worldforge.time.calendar import CalendarClock, parse_duration
from worldforge.time.event_driven import EventDrivenClock


# ============================================================
# parse_duration
# ============================================================

class TestParseDuration:
    def test_day(self):
        assert parse_duration("1 day") == timedelta(days=1)

    def test_days_plural(self):
        assert parse_duration("7 days") == timedelta(days=7)

    def test_week(self):
        assert parse_duration("1 week") == timedelta(weeks=1)

    def test_hour(self):
        assert parse_duration("1 hour") == timedelta(hours=1)

    def test_minute(self):
        assert parse_duration("30 minutes") == timedelta(minutes=30)

    def test_second(self):
        assert parse_duration("10 seconds") == timedelta(seconds=10)

    def test_timedelta_passthrough(self):
        td = timedelta(days=3)
        assert parse_duration(td) is td

    def test_unknown_unit_raises(self):
        with pytest.raises(ConfigurationError):
            parse_duration("5 fortnights")

    def test_missing_unit_raises(self):
        with pytest.raises(ConfigurationError):
            parse_duration("5")


# ============================================================
# CalendarClock
# ============================================================

class TestCalendarClock:
    def test_now_starts_at_start(self):
        clock = CalendarClock(
            start="2024-01-01", end="2024-12-31", step="1 day"
        )
        assert clock.now == datetime(2024, 1, 1)

    def test_tick_advances_by_step(self):
        clock = CalendarClock(
            start="2024-01-01", end="2024-12-31", step="1 day"
        )
        clock.tick()
        assert clock.now == datetime(2024, 1, 2)

    def test_tick_hourly(self):
        clock = CalendarClock(
            start="2024-01-01", end="2024-01-02", step="1 hour"
        )
        for _ in range(3):
            clock.tick()
        assert clock.now == datetime(2024, 1, 1, 3, 0, 0)

    def test_is_done_false_before_end(self):
        clock = CalendarClock(
            start="2024-01-01", end="2024-12-31", step="1 day"
        )
        assert not clock.is_done

    def test_is_done_true_at_end(self):
        # start=Jan 1, end=Jan 3, step=1 day
        # tick → Jan 2 (< Jan 3, not done)
        # tick → Jan 3 (== Jan 3, done)
        clock = CalendarClock(
            start="2024-01-01", end="2024-01-03", step="1 day"
        )
        clock.tick()  # Jan 2
        assert not clock.is_done
        clock.tick()  # Jan 3 >= end → done
        assert clock.is_done

    def test_reset(self):
        clock = CalendarClock(
            start="2024-01-01", end="2024-12-31", step="1 day"
        )
        clock.tick()
        clock.tick()
        clock.reset()
        assert clock.now == datetime(2024, 1, 1)
        assert not clock.is_done

    def test_step_property(self):
        clock = CalendarClock(
            start="2024-01-01", end="2024-12-31", step="1 week"
        )
        assert clock.step == timedelta(weeks=1)

    def test_start_equals_end_raises(self):
        with pytest.raises(ConfigurationError):
            CalendarClock(
                start="2024-01-01", end="2024-01-01", step="1 day"
            )

    def test_start_after_end_raises(self):
        with pytest.raises(ConfigurationError):
            CalendarClock(
                start="2024-12-31", end="2024-01-01", step="1 day"
            )

    def test_datetime_objects_accepted(self):
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 30)
        clock = CalendarClock(start=start, end=end, step=timedelta(days=1))
        assert clock.now == start

    def test_total_steps_calculation(self):
        """7-day clock with 1-day step → 7 ticks to reach end."""
        clock = CalendarClock(
            start="2024-01-01", end="2024-01-08", step="1 day"
        )
        ticks = 0
        while not clock.is_done:
            clock.tick()
            ticks += 1
        assert ticks == 7


# ============================================================
# EventDrivenClock
# ============================================================

class TestEventDrivenClock:
    def test_initial_now_is_zero(self):
        clock = EventDrivenClock(max_time=1000)
        assert clock.now == 0.0

    def test_advance_to(self):
        clock = EventDrivenClock(max_time=1000)
        clock.advance_to(42.5)
        assert clock.now == 42.5

    def test_advance_to_same_time(self):
        clock = EventDrivenClock(max_time=1000)
        clock.advance_to(10.0)
        clock.advance_to(10.0)   # same time is fine
        assert clock.now == 10.0

    def test_backward_advance_raises(self):
        clock = EventDrivenClock(max_time=1000)
        clock.advance_to(50.0)
        with pytest.raises(EventOrderError):
            clock.advance_to(10.0)

    def test_is_done_before_max(self):
        clock = EventDrivenClock(max_time=100)
        clock.advance_to(50)
        assert not clock.is_done

    def test_is_done_at_max(self):
        clock = EventDrivenClock(max_time=100)
        clock.advance_to(100)
        assert clock.is_done

    def test_tick_is_noop(self):
        clock = EventDrivenClock(max_time=100)
        clock.advance_to(5.0)
        clock.tick()   # should do nothing
        assert clock.now == 5.0

    def test_reset(self):
        clock = EventDrivenClock(max_time=100)
        clock.advance_to(75.0)
        clock.reset()
        assert clock.now == 0.0
        assert not clock.is_done

    def test_max_time_zero_raises(self):
        with pytest.raises(ConfigurationError):
            EventDrivenClock(max_time=0)

    def test_min_step_tolerance(self):
        """Tiny backward moves within min_step tolerance are allowed."""
        clock = EventDrivenClock(max_time=1000, min_step=1e-9)
        clock.advance_to(10.0)
        # Advance by exactly the tolerance — should not raise
        clock.advance_to(10.0 - 1e-10)
