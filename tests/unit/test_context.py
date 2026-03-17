"""Tests for SimContext (Step 09)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from worldforge.agent import Agent, field, _reset_id_counter
from worldforge.core.clock import DiscreteClock
from worldforge.core.context import SimContext
from worldforge.events.base import Event


@dataclass
class SampleEvent(Event):
    value: float


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


@pytest.fixture
def clock():
    return DiscreteClock(steps=100)


@pytest.fixture
def ctx(clock, rng):
    return SimContext(clock=clock, rng=rng)


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------

class TestAgentRegistration:
    def test_register_and_count(self, ctx, rng):
        a = Agent(_rng=rng)
        ctx._register_agent(a)
        assert ctx.agent_count() == 1

    def test_unregister(self, ctx, rng):
        a = Agent(_rng=rng)
        ctx._register_agent(a)
        ctx._unregister_agent(a)
        assert ctx.agent_count() == 0

    def test_get_agent_by_id(self, ctx, rng):
        a = Agent(_rng=rng)
        ctx._register_agent(a)
        assert ctx.get_agent(a.id) is a

    def test_get_nonexistent_agent(self, ctx):
        assert ctx.get_agent("nonexistent") is None

    def test_agents_by_type(self, ctx, rng):
        class UserAgent(Agent):
            pass

        class BotAgent(Agent):
            pass

        u = UserAgent(_rng=rng)
        b = BotAgent(_rng=rng)
        ctx._register_agent(u)
        ctx._register_agent(b)

        users = ctx.agents(UserAgent)
        bots = ctx.agents(BotAgent)
        all_agents = ctx.agents()

        assert len(users) == 1 and users[0] is u
        assert len(bots) == 1 and bots[0] is b
        assert len(all_agents) == 2

    def test_agents_with_filter(self, ctx, rng):
        class A(Agent):
            score: int = field(0)

        for i in range(5):
            ctx._register_agent(A(_rng=rng, score=i))

        high = ctx.agents(A, filter=lambda a: a.score >= 3)
        assert len(high) == 2
        assert all(a.score >= 3 for a in high)

    def test_agent_count_by_type(self, ctx, rng):
        class X(Agent): pass
        class Y(Agent): pass

        for _ in range(3):
            ctx._register_agent(X(_rng=rng))
        for _ in range(2):
            ctx._register_agent(Y(_rng=rng))

        assert ctx.agent_count(X) == 3
        assert ctx.agent_count(Y) == 2
        assert ctx.agent_count() == 5


# ---------------------------------------------------------------------------
# Agent statistics
# ---------------------------------------------------------------------------

class TestAgentStats:
    def test_agent_mean(self, ctx, rng):
        class A(Agent):
            value: float = field(0.0)

        for v in [1.0, 2.0, 3.0]:
            ctx._register_agent(A(_rng=rng, value=v))

        assert abs(ctx.agent_mean(A, "value") - 2.0) < 1e-9

    def test_agent_mean_empty(self, ctx):
        class A(Agent): pass
        assert ctx.agent_mean(A, "value") == 0.0

    def test_agent_percentile(self, ctx, rng):
        class A(Agent):
            v: float = field(0.0)

        for val in range(10):
            ctx._register_agent(A(_rng=rng, v=float(val)))

        p50 = ctx.agent_percentile(A, "v", 0.5)
        assert p50 == 5.0


# ---------------------------------------------------------------------------
# Emit and event log
# ---------------------------------------------------------------------------

class TestEmitAndEventLog:
    def test_emit_sets_timestamp(self, ctx, clock):
        clock.tick()
        e = SampleEvent(value=42.0)
        ctx.emit(e)
        assert e.timestamp == 1

    def test_emit_adds_to_log(self, ctx):
        e = SampleEvent(value=1.0)
        ctx.emit(e)
        assert e in ctx._event_log

    def test_event_count(self, ctx):
        for v in range(5):
            ctx.emit(SampleEvent(value=float(v)))
        assert ctx.event_count(SampleEvent) == 5

    def test_event_sum(self, ctx):
        for v in [10.0, 20.0, 30.0]:
            ctx.emit(SampleEvent(value=v))
        assert abs(ctx.event_sum(SampleEvent, "value") - 60.0) < 1e-9

    def test_event_handler_called(self, ctx):
        received = []
        ctx.register_event_handler(SampleEvent, lambda e, c: received.append(e.value))
        ctx.emit(SampleEvent(value=99.0))
        assert received == [99.0]

    def test_on_event_dispatched_to_overriding_agents(self, ctx, rng):
        received = []

        class Listener(Agent):
            def on_event(self, event, ctx):
                if isinstance(event, SampleEvent):
                    received.append(event.value)

        ctx._register_agent(Listener(_rng=rng))
        ctx.emit(SampleEvent(value=7.0))
        assert received == [7.0]

    def test_on_event_not_dispatched_to_base_agent(self, ctx, rng):
        """Base Agent.on_event is a no-op; the runner skips it for performance."""
        called = []
        original = Agent.on_event
        Agent.on_event = lambda self, e, c: called.append(True)
        try:
            ctx._register_agent(Agent(_rng=rng))
            ctx.emit(SampleEvent(value=1.0))
        finally:
            Agent.on_event = original
        assert called == []


# ---------------------------------------------------------------------------
# Deferred mutations
# ---------------------------------------------------------------------------

class TestDeferredMutations:
    def test_remove_agent_deferred(self, ctx, rng):
        a = Agent(_rng=rng)
        ctx._register_agent(a)
        ctx.remove_agent(a)
        assert ctx.agent_count() == 1   # not yet
        ctx._flush_pending()
        assert ctx.agent_count() == 0   # now gone

    def test_remove_agent_calls_on_die(self, ctx, rng):
        log = []

        class A(Agent):
            def on_die(self, ctx):
                log.append("died")

        a = A(_rng=rng)
        ctx._register_agent(a)
        ctx.remove_agent(a)
        ctx._flush_pending()
        assert log == ["died"]

    def test_spawn_agent(self, ctx):
        class NewAgent(Agent):
            pass

        ctx.spawn(NewAgent, count=3)
        ctx._flush_pending()
        assert ctx.agent_count(NewAgent) == 3

    def test_spawn_calls_on_born(self, ctx):
        log = []

        class A(Agent):
            def on_born(self, ctx):
                log.append("born")

        ctx.spawn(A, count=2)
        ctx._flush_pending()
        assert log == ["born", "born"]

    def test_spawn_with_init(self, ctx):
        class A(Agent):
            score: float = field(0.0)

        ctx.spawn(A, count=3, init=lambda a: setattr(a, "score", 99.0))
        ctx._flush_pending()
        for a in ctx.agents(A):
            assert a.score == 99.0

    def test_remove_during_step_is_safe(self, rng):
        """Agents removing themselves during step() must not corrupt iteration."""
        clock = DiscreteClock(steps=1)
        ctx = SimContext(clock=clock, rng=rng)

        class SelfRemover(Agent):
            def step(self, ctx):
                ctx.remove_agent(self)

        for _ in range(5):
            ctx._register_agent(SelfRemover(_rng=rng))

        ctx._run_tick()
        assert ctx.agent_count() == 0


# ---------------------------------------------------------------------------
# Full tick
# ---------------------------------------------------------------------------

class TestRunTick:
    def test_step_called_on_all_agents(self, rng):
        clock = DiscreteClock(steps=3)
        ctx = SimContext(clock=clock, rng=rng)
        step_counts = {}

        class A(Agent):
            def step(self, ctx):
                step_counts[self.id] = step_counts.get(self.id, 0) + 1

        for _ in range(3):
            ctx._register_agent(A(_rng=rng))

        ctx._run_tick()
        assert all(v == 1 for v in step_counts.values())
        assert len(step_counts) == 3

    def test_ctx_cleared_after_step(self, rng):
        clock = DiscreteClock(steps=1)
        ctx = SimContext(clock=clock, rng=rng)

        class A(Agent):
            def step(self, ctx_inner):
                pass

        a = A(_rng=rng)
        ctx._register_agent(a)
        ctx._run_tick()
        assert a._ctx is None
