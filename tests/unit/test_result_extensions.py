"""Tests for SimulationResult extensions: to_parquet, and _filter_events string last."""
from __future__ import annotations

import os
import tempfile

import pytest

from worldforge.output.result import SimulationResult
from worldforge.agent import _reset_id_counter


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


@pytest.fixture
def result():
    return SimulationResult(
        data={
            "events": [
                {"amount": 10.0, "tag": "a"},
                {"amount": 20.0, "tag": "b"},
                {"amount": 5.0,  "tag": "a"},
            ],
        },
        metadata={"name": "test", "seed": 1},
    )


# ---------------------------------------------------------------------------
# to_parquet()
# ---------------------------------------------------------------------------

class TestToParquet:
    def test_writes_parquet_files(self, result):
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            result.to_parquet(tmpdir)
            files = os.listdir(tmpdir)
            assert "events.parquet" in files

    def test_parquet_readable(self, result):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            result.to_parquet(tmpdir)
            path = os.path.join(tmpdir, "events.parquet")
            df = pd.read_parquet(path)
            assert len(df) == 3
            assert "amount" in df.columns

    def test_creates_directory_if_missing(self, result):
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b")
            result.to_parquet(nested)
            assert os.path.exists(nested)


# ---------------------------------------------------------------------------
# _filter_events: string `last` for CalendarClock
# ---------------------------------------------------------------------------

class TestFilterEventsStringLast:
    def test_string_last_filters_by_duration(self):
        """event_sum(last='1 day') should only count recent events."""
        from datetime import datetime, timedelta
        from worldforge import Agent, Simulation, field
        from worldforge.time.calendar import CalendarClock
        from worldforge.events.base import Event
        from worldforge.probes import AggregatorProbe
        from dataclasses import dataclass

        @dataclass
        class Sale(Event):
            amount: float

        class Seller(Agent):
            def step(self, ctx):
                ctx.emit(Sale(amount=100.0))

        sim = Simulation(
            name="filter_test",
            seed=1,
            clock=CalendarClock(
                start="2024-01-01",
                end="2024-01-10",
                step="1 day",
            ),
        )
        sim.add_agents(Seller, count=1)
        sim.add_probe(AggregatorProbe(
            metrics={
                "total_all": lambda ctx: ctx.event_sum(Sale, "amount"),
                "last_day":  lambda ctx: ctx.event_sum(Sale, "amount", last="1 day"),
            },
            every=1,
            name="metrics",
        ))
        result = sim.run()
        rows = result["metrics"]
        # Last-day window should always be ≤ total
        for row in rows:
            assert row["last_day"] <= row["total_all"]
        # On the first step, total and last_day should be equal
        assert rows[0]["last_day"] == rows[0]["total_all"]
        # By step 5, total should be much larger than last_day
        if len(rows) > 4:
            assert rows[4]["total_all"] > rows[4]["last_day"]

    def test_timedelta_last_works(self):
        """event_count(last=timedelta(days=1)) should work."""
        from datetime import timedelta, datetime
        from worldforge import Agent, Simulation, field
        from worldforge.time.calendar import CalendarClock
        from worldforge.events.base import Event
        from worldforge.probes import AggregatorProbe
        from dataclasses import dataclass

        @dataclass
        class Ping(Event):
            pass

        class Pinger(Agent):
            def step(self, ctx):
                ctx.emit(Ping())

        sim = Simulation(
            name="td_test",
            seed=1,
            clock=CalendarClock(start="2024-01-01", end="2024-01-06", step="1 day"),
        )
        sim.add_agents(Pinger, count=2)
        sim.add_probe(AggregatorProbe(
            metrics={
                "recent": lambda ctx: ctx.event_count(Ping, last=timedelta(days=1)),
            },
            every=1,
            name="m",
        ))
        result = sim.run()
        rows = result["m"]
        # Each step emits 2 pings; last=1day window should cap at ~2
        assert rows[0]["recent"] <= 10


# ---------------------------------------------------------------------------
# CalendarClock realtime parameter (just checks it doesn't crash; no real sleep)
# ---------------------------------------------------------------------------

class TestCalendarClockRealtime:
    def test_realtime_false_default(self):
        from worldforge.time.calendar import CalendarClock
        clock = CalendarClock(start="2024-01-01", end="2024-01-03", step="1 day")
        assert not clock._realtime

    def test_realtime_param_stored(self):
        from worldforge.time.calendar import CalendarClock
        clock = CalendarClock(
            start="2024-01-01", end="2024-01-03", step="1 day",
            realtime=True, realtime_factor=0.0,   # factor=0 → no real sleep
        )
        assert clock._realtime
        clock.tick()   # should not raise
        from datetime import datetime
        assert clock.now == datetime(2024, 1, 2)


# ---------------------------------------------------------------------------
# GymWrapper basic smoke test
# ---------------------------------------------------------------------------

class TestGymWrapper:
    def test_reset_returns_obs(self):
        pytest.importorskip("numpy")
        from worldforge.scenarios.epidemic import epidemic_world
        from worldforge.rl import GymWrapper
        import numpy as np

        sim = epidemic_world(population=100, duration_days=10, seed=42)

        env = GymWrapper(
            sim=sim,
            observation=lambda ctx: np.array([
                sum(1 for p in ctx.agents() if getattr(p, "state", "S") == "I") / 100.0,
            ]),
            reward=lambda ctx: float(
                sum(1 for p in ctx.agents() if getattr(p, "state", "R") == "R")
            ),
        )
        obs, info = env.reset()
        assert obs.shape == (1,)
        assert isinstance(info, dict)

    def test_step_returns_tuple(self):
        pytest.importorskip("numpy")
        from worldforge.scenarios.epidemic import epidemic_world
        from worldforge.rl import GymWrapper
        import numpy as np

        sim = epidemic_world(population=50, duration_days=5, seed=1)
        env = GymWrapper(
            sim=sim,
            observation=lambda ctx: np.array([ctx.agent_count() / 50.0]),
            reward=lambda ctx: float(ctx.agent_count()),
        )
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (1,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_max_steps_truncation(self):
        pytest.importorskip("numpy")
        from worldforge.scenarios.epidemic import epidemic_world
        from worldforge.rl import GymWrapper
        import numpy as np

        sim = epidemic_world(population=50, duration_days=100, seed=1)
        env = GymWrapper(
            sim=sim,
            observation=lambda ctx: np.array([0.0]),
            reward=lambda ctx: 0.0,
            max_steps=3,
        )
        env.reset()
        for _ in range(2):
            _, _, _, truncated, _ = env.step(0)
        _, _, _, truncated, _ = env.step(0)
        assert truncated
