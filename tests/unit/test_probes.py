"""Unit tests for all probe classes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from worldforge.agent import Agent, field, _reset_id_counter
from worldforge.core.clock import DiscreteClock
from worldforge.core.context import SimContext
from worldforge.events.base import Event
from worldforge.probes import (
    EventLogProbe, SnapshotProbe, AggregatorProbe,
    TimeSeriesProbe, CustomProbe,
)
from worldforge.probes.base import Probe, _resolve_every


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clock():
    return DiscreteClock(steps=100)


@pytest.fixture
def ctx(clock, rng):
    return SimContext(clock=clock, rng=rng)


@dataclass
class SaleEvent(Event):
    amount: float


@dataclass
class OtherEvent(Event):
    info: str


class ScoredAgent(Agent):
    score: float = field(0.0)
    tier: str = field("free")


# ============================================================
# _resolve_every
# ============================================================

class TestResolveEvery:
    def test_int_passthrough(self, clock):
        assert _resolve_every(5, clock) == 5
        assert _resolve_every(1, clock) == 1

    def test_zero_clamped_to_1(self, clock):
        assert _resolve_every(0, clock) == 1

    def test_string_integer_prefix(self, clock):
        # DiscreteClock: parse the integer prefix
        assert _resolve_every("7 steps", clock) == 7

    def test_string_calendar_day(self):
        from worldforge.time.calendar import CalendarClock
        c = CalendarClock(start="2024-01-01", end="2025-01-01", step="1 day")
        assert _resolve_every("7 days", c) == 7
        assert _resolve_every("1 week", c) == 7

    def test_unknown_falls_back_to_1(self, clock):
        assert _resolve_every(None, clock) == 1


# ============================================================
# Probe.on_step interval gating
# ============================================================

class TestProbeInterval:
    def test_collects_at_interval(self, ctx):
        collected = []

        class TrackProbe(Probe):
            def collect(self, c):
                collected.append(c.now)

            def finalize(self):
                return collected

        p = TrackProbe(every=3, name="track")
        p.configure(DiscreteClock(steps=10))
        for step in range(1, 11):
            p.on_step(ctx, step)

        # Should fire at steps 3, 6, 9
        assert collected == [0, 0, 0]  # ctx.now doesn't change (fixture clock at 0)
        assert len(collected) == 3

    def test_every_1_collects_each_step(self, ctx):
        counts = []

        class CountProbe(Probe):
            def collect(self, c): counts.append(1)
            def finalize(self): return counts

        p = CountProbe(every=1, name="count")
        p.configure(DiscreteClock(steps=5))
        for step in range(1, 6):
            p.on_step(ctx, step)
        assert len(counts) == 5


# ============================================================
# EventLogProbe
# ============================================================

class TestEventLogProbe:
    def test_collects_matching_events(self, ctx, clock):
        probe = EventLogProbe(events=[SaleEvent], name="sales")
        probe.configure(clock)
        clock.tick()
        ctx.emit(SaleEvent(amount=50.0))
        ctx.emit(OtherEvent(info="ignored"))
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert len(records) == 1
        assert records[0]["amount"] == 50.0

    def test_filters_by_type(self, ctx, clock):
        probe = EventLogProbe(events=[OtherEvent], name="other")
        probe.configure(clock)
        clock.tick()
        ctx.emit(SaleEvent(amount=99.0))
        ctx.emit(OtherEvent(info="hello"))
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert len(records) == 1
        assert records[0]["info"] == "hello"

    def test_incremental_collection(self, ctx, clock):
        """_last_idx prevents re-reading already-collected events."""
        probe = EventLogProbe(events=[SaleEvent], name="inc")
        probe.configure(clock)

        clock.tick()
        ctx.emit(SaleEvent(amount=10.0))
        probe.on_step(ctx, 1)

        clock.tick()
        ctx.emit(SaleEvent(amount=20.0))
        probe.on_step(ctx, 2)

        records = probe.finalize()
        assert len(records) == 2
        assert records[0]["amount"] == 10.0
        assert records[1]["amount"] == 20.0

    def test_private_fields_excluded(self, ctx, clock):
        """Fields starting with _ should not appear in records."""

        @dataclass
        class PrivateEvent(Event):
            public_val: float
            _private: str = "_hidden"

        probe = EventLogProbe(events=[PrivateEvent], name="priv")
        probe.configure(clock)
        clock.tick()
        ctx.emit(PrivateEvent(public_val=1.0))
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert len(records) == 1
        assert "_private" not in records[0]
        assert "public_val" in records[0]

    def test_finalize_returns_copy(self, ctx, clock):
        probe = EventLogProbe(events=[SaleEvent], name="copy")
        probe.configure(clock)
        r1 = probe.finalize()
        r2 = probe.finalize()
        assert r1 is not r2


# ============================================================
# SnapshotProbe
# ============================================================

class TestSnapshotProbe:
    def _populate(self, ctx, n=5):
        for i in range(n):
            a = ScoredAgent(score=float(i), tier="pro" if i % 2 == 0 else "free")
            ctx._register_agent(a)

    def test_collects_specified_fields(self, ctx, clock):
        probe = SnapshotProbe(
            agent_type=ScoredAgent,
            fields=["score", "tier"],
            every=1,
            name="snap",
        )
        probe.configure(clock)
        self._populate(ctx)
        clock.tick()
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert len(records) == 5
        for r in records:
            assert "score" in r
            assert "tier" in r
            assert "timestamp" in r

    def test_sample_rate(self, ctx, clock):
        probe = SnapshotProbe(
            agent_type=ScoredAgent,
            fields=["score"],
            every=1,
            sample_rate=0.4,
            name="sampled",
        )
        probe.configure(clock)
        for i in range(20):
            ctx._register_agent(ScoredAgent(score=float(i)))
        clock.tick()
        probe.on_step(ctx, 1)
        records = probe.finalize()
        # sample_rate=0.4 of 20 → 8 agents; allow ±2 for rounding
        assert 6 <= len(records) <= 10

    def test_missing_field_returns_none(self, ctx, clock):
        probe = SnapshotProbe(
            agent_type=ScoredAgent,
            fields=["score", "nonexistent_field"],
            every=1,
            name="missing",
        )
        probe.configure(clock)
        ctx._register_agent(ScoredAgent(score=1.0))
        clock.tick()
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert records[0]["nonexistent_field"] is None

    def test_timestamp_is_correct(self, clock, rng):
        ctx2 = SimContext(clock=clock, rng=rng)
        probe = SnapshotProbe(
            agent_type=ScoredAgent,
            fields=["score"],
            every=1,
            name="ts",
        )
        probe.configure(clock)
        ctx2._register_agent(ScoredAgent(score=5.0))
        clock.tick()  # now = 1
        probe.on_step(ctx2, 1)
        records = probe.finalize()
        assert records[0]["timestamp"] == 1


# ============================================================
# AggregatorProbe
# ============================================================

class TestAggregatorProbe:
    def test_computes_metrics(self, ctx, clock):
        probe = AggregatorProbe(
            metrics={
                "count": lambda c: c.agent_count(ScoredAgent),
                "total": lambda c: sum(a.score for a in c.agents(ScoredAgent)),
            },
            every=1,
            name="agg",
        )
        probe.configure(clock)
        for i in range(3):
            ctx._register_agent(ScoredAgent(score=float(i * 10)))
        clock.tick()
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert len(records) == 1
        assert records[0]["count"] == 3
        assert abs(records[0]["total"] - 30.0) < 1e-9

    def test_metric_exception_returns_none(self, ctx, clock):
        probe = AggregatorProbe(
            metrics={"bad": lambda c: 1 / 0},
            every=1,
            name="bad",
        )
        probe.configure(clock)
        clock.tick()
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert records[0]["bad"] is None

    def test_timestamp_included(self, ctx, clock):
        probe = AggregatorProbe(
            metrics={"n": lambda c: 0},
            every=1,
            name="ts",
        )
        probe.configure(clock)
        clock.tick()
        probe.on_step(ctx, 1)
        assert "timestamp" in probe.finalize()[0]

    def test_multiple_collections(self, ctx, clock):
        probe = AggregatorProbe(
            metrics={"n": lambda c: c.agent_count()},
            every=1,
            name="multi",
        )
        probe.configure(clock)
        ctx._register_agent(ScoredAgent())
        for step in range(1, 4):
            clock.tick()
            probe.on_step(ctx, step)
        records = probe.finalize()
        assert len(records) == 3


# ============================================================
# TimeSeriesProbe
# ============================================================

class TestTimeSeriesProbe:
    def test_collects_series(self, ctx, clock):
        probe = TimeSeriesProbe(
            series={"avg_score": lambda c: c.agent_mean(ScoredAgent, "score")},
            every=1,
            name="ts",
        )
        probe.configure(clock)
        for i in range(4):
            ctx._register_agent(ScoredAgent(score=float(i)))
        clock.tick()
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert len(records) == 1
        assert abs(records[0]["avg_score"] - 1.5) < 1e-9
        assert "timestamp" in records[0]

    def test_exception_in_series_returns_none(self, ctx, clock):
        probe = TimeSeriesProbe(
            series={"bad": lambda c: 1 / 0},
            every=1,
            name="bad",
        )
        probe.configure(clock)
        clock.tick()
        probe.on_step(ctx, 1)
        assert probe.finalize()[0]["bad"] is None

    def test_interval_filtering(self, ctx, clock):
        probe = TimeSeriesProbe(
            series={"n": lambda c: c.agent_count()},
            every=5,
            name="every5",
        )
        probe.configure(clock)
        for step in range(1, 11):
            clock.tick()
            probe.on_step(ctx, step)
        # every=5 → collected at steps 5 and 10
        records = probe.finalize()
        assert len(records) == 2


# ============================================================
# CustomProbe
# ============================================================

class TestCustomProbe:
    def test_collector_record(self, ctx, clock):
        def my_fn(c, collector):
            collector.record({"step": c.now, "n": c.agent_count()})

        probe = CustomProbe(fn=my_fn, every=1, name="custom")
        probe.configure(clock)
        clock.tick()
        probe.on_step(ctx, 1)
        records = probe.finalize()
        assert len(records) == 1
        assert "step" in records[0]
        assert "n" in records[0]

    def test_multiple_records_per_collect(self, ctx, clock):
        def multi_fn(c, collector):
            for i in range(3):
                collector.record({"item": i})

        probe = CustomProbe(fn=multi_fn, every=1, name="multi")
        probe.configure(clock)
        clock.tick()
        probe.on_step(ctx, 1)
        assert len(probe.finalize()) == 3

    def test_name_from_function(self):
        def my_probe_fn(c, col): pass
        p = CustomProbe(fn=my_probe_fn, every=1)
        assert p.name == "my_probe_fn"
