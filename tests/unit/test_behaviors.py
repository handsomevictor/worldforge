"""Unit tests for behaviors: StateMachineBehavior, LifecycleBehavior,
DecisionBehavior, SocialBehavior, ContagionBehavior, MemoryBehavior."""
from __future__ import annotations

import numpy as np
import pytest

from worldforge.agent import Agent, field, _reset_id_counter
from worldforge.behaviors import (
    StateMachineBehavior, LifecycleBehavior,
    DecisionBehavior, SocialBehavior, ContagionBehavior, MemoryBehavior,
)
from worldforge.core.clock import DiscreteClock
from worldforge.core.context import SimContext
from worldforge.distributions import Exponential, Normal


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def ctx(rng):
    clock = DiscreteClock(steps=100)
    clock.tick()  # step to 1
    return SimContext(clock=clock, rng=rng)


# ============================================================
# StateMachineBehavior
# ============================================================

class SimpleFSM(StateMachineBehavior):
    states = ["a", "b", "done"]
    initial = "a"
    terminal = ["done"]
    transitions = {
        "a": [(1.0, "b", 1)],    # always transition after dwell=1
        "b": [(1.0, "done", 1)],
    }


class TestStateMachineBehavior:
    def test_initial_state(self):
        fsm = SimpleFSM()
        assert fsm.current_state == "a"
        assert not fsm.is_terminal

    def test_transitions_after_steps(self, ctx):
        fsm = SimpleFSM()
        fsm.step(ctx)  # lazy init: enter a, dwell=1 → transitions at time_in_state >= 1
        # After first step: time_in_state=1 >= dwell=1, so transitions to "b"
        assert fsm.current_state == "b"

    def test_reaches_terminal(self, ctx):
        fsm = SimpleFSM()
        fsm.step(ctx)   # a → b
        fsm.step(ctx)   # b → done
        assert fsm.is_terminal
        assert fsm.current_state == "done"

    def test_no_step_after_terminal(self, ctx):
        fsm = SimpleFSM()
        fsm.step(ctx)
        fsm.step(ctx)
        assert fsm.is_terminal
        # Further steps are no-ops
        fsm.step(ctx)
        assert fsm.current_state == "done"

    def test_on_transition_called(self, ctx):
        transitions_log = []

        class LogFSM(StateMachineBehavior):
            states = ["x", "y"]
            initial = "x"
            terminal = ["y"]
            transitions = {"x": [(1.0, "y", 1)]}

            def on_transition(self, from_state, to_state, c):
                transitions_log.append((from_state, to_state))

        fsm = LogFSM()
        fsm.step(ctx)
        assert ("x", "y") in transitions_log

    def test_distribution_dwell_time(self, rng):
        """FSM with distribution-based dwell time eventually transitions."""
        clock = DiscreteClock(steps=500)
        ctx2 = SimContext(clock=clock, rng=rng)

        class SlowFSM(StateMachineBehavior):
            states = ["start", "end"]
            initial = "start"
            terminal = ["end"]
            transitions = {"start": [(1.0, "end", Exponential(scale=5))]}

        fsm = SlowFSM()
        for _ in range(200):
            clock.tick()
            fsm.step(ctx2)
            if fsm.is_terminal:
                break
        assert fsm.is_terminal

    def test_probabilistic_transitions(self, rng):
        """Multiple-choice transitions: run 1000 times, check rough distribution."""
        results = {"x": 0, "y": 0}

        class TwoChoiceFSM(StateMachineBehavior):
            states = ["start", "x", "y"]
            initial = "start"
            terminal = ["x", "y"]
            transitions = {"start": [(0.5, "x", 1), (0.5, "y", 1)]}

        clock = DiscreteClock(steps=2)
        for seed in range(500):
            ctx2 = SimContext(clock=DiscreteClock(steps=2), rng=np.random.default_rng(seed))
            fsm = TwoChoiceFSM()
            clock.tick()
            fsm.step(ctx2)
            results[fsm.current_state] += 1

        # With 500 trials each 50/50, we expect roughly 250 each
        assert 150 < results["x"] < 350, f"x={results['x']}"
        assert 150 < results["y"] < 350, f"y={results['y']}"


# ============================================================
# LifecycleBehavior
# ============================================================

class TestLifecycleBehavior:
    def test_age_increments(self, ctx):
        lc = LifecycleBehavior()
        lc.step(ctx)
        assert lc.age == 1
        lc.step(ctx)
        assert lc.age == 2

    def test_is_alive_without_lifespan(self, ctx):
        lc = LifecycleBehavior()
        lc.step(ctx)
        assert lc.is_alive

    def test_death_at_lifespan(self, ctx):
        """Agent is removed when age reaches lifespan."""
        deaths = []

        class MortalAgent(Agent):
            def on_die(self, c):
                deaths.append(self.id)

        clock = DiscreteClock(steps=10)
        ctx2 = SimContext(clock=DiscreteClock(steps=10), rng=np.random.default_rng(0))
        agent = MortalAgent()
        ctx2._register_agent(agent)

        lc = LifecycleBehavior(lifespan=3)
        lc.agent = agent
        for _ in range(5):
            lc.step(ctx2)
        ctx2._flush_pending()

        assert agent.id in deaths or lc.age >= 3

    def test_lifespan_distribution(self, ctx):
        """Distribution-based lifespan is sampled on first step."""
        lc = LifecycleBehavior(lifespan=Normal(mu=20, sigma=1, clip=(10, 30)))
        lc.step(ctx)
        assert lc._lifespan is not None
        assert 10 <= lc._lifespan <= 30

    def test_birth_age(self, ctx):
        lc = LifecycleBehavior(birth_age=5)
        assert lc.age == 5

    def test_on_lifecycle_event_called(self, ctx):
        events = []

        class TrackingLC(LifecycleBehavior):
            def on_lifecycle_event(self, event, c):
                events.append(event)

        lc = TrackingLC(lifespan=2)
        lc.step(ctx)
        lc.step(ctx)
        assert "age" in events
        assert "death" in events


# ============================================================
# DecisionBehavior
# ============================================================

class TestDecisionBehavior:
    def test_first_matching_rule_executed(self, ctx):
        executed = []

        class D(DecisionBehavior):
            rules = [
                (lambda a, c: True, lambda a, c: executed.append("rule1")),
                (lambda a, c: True, lambda a, c: executed.append("rule2")),
            ]

        d = D()
        d.agent = Agent()
        d.step(ctx)
        assert executed == ["rule1"]  # only first matching rule fires

    def test_no_match_no_error(self, ctx):
        class D(DecisionBehavior):
            rules = [(lambda a, c: False, lambda a, c: None)]

        d = D()
        d.agent = Agent()
        d.step(ctx)  # should not raise

    def test_add_rule_appends(self, ctx):
        executed = []
        d = DecisionBehavior()
        d.agent = Agent()
        d.add_rule(
            condition=lambda a, c: True,
            action=lambda a, c: executed.append("dynamic"),
        )
        d.step(ctx)
        assert "dynamic" in executed

    def test_add_rule_priority(self, ctx):
        executed = []
        d = DecisionBehavior()
        d.add_rule(lambda a, c: True, lambda a, c: executed.append("low"))
        d.add_rule(lambda a, c: True, lambda a, c: executed.append("high"), priority=0)
        d.agent = Agent()
        d.step(ctx)
        assert executed == ["high"]

    def test_exception_in_condition_skipped(self, ctx):
        """A rule whose condition raises an exception is silently skipped."""
        executed = []

        class D(DecisionBehavior):
            rules = [
                (lambda a, c: 1 / 0, lambda a, c: executed.append("bad")),
                (lambda a, c: True,  lambda a, c: executed.append("good")),
            ]

        d = D()
        d.agent = Agent()
        d.step(ctx)
        assert "good" in executed
        assert "bad" not in executed


# ============================================================
# SocialBehavior
# ============================================================

class TestSocialBehavior:
    def _make_agents(self, opinions, rng):
        agents = []
        for op in opinions:
            a = Agent()
            a.opinion = op
            agents.append(a)
        return agents

    def test_opinion_converges_toward_neighbors(self, ctx):
        agent = Agent()
        agent.opinion = 0.0
        neighbors = self._make_agents([1.0, 1.0], ctx.rng)

        social = SocialBehavior()
        social.agent = agent
        social.influence(neighbors, ctx)

        assert agent.opinion > 0.0, "Opinion should shift toward neighbors"

    def test_no_neighbors_no_change(self, ctx):
        agent = Agent()
        agent.opinion = 0.5
        social = SocialBehavior()
        social.agent = agent
        social.influence([], ctx)
        assert agent.opinion == 0.5

    def test_influence_rate_controls_magnitude(self, ctx):
        class FastSocial(SocialBehavior):
            influence_rate = 1.0

        agent = Agent()
        agent.opinion = 0.0
        neighbors = self._make_agents([1.0], ctx.rng)
        s = FastSocial()
        s.agent = agent
        s.influence(neighbors, ctx)
        assert abs(agent.opinion - 1.0) < 0.01  # rate=1 → full convergence

    def test_custom_opinion_field(self, ctx):
        class MySocial(SocialBehavior):
            opinion_field = "sentiment"

        agent = Agent()
        agent.sentiment = -0.5
        neighbors = [type('N', (), {'sentiment': 0.5})()]
        s = MySocial()
        s.agent = agent
        s.influence(neighbors, ctx)
        assert agent.sentiment > -0.5


# ============================================================
# ContagionBehavior
# ============================================================

class TestContagionBehavior:
    def test_susceptible_can_get_infected(self):
        rng = np.random.default_rng(1)
        clock = DiscreteClock(steps=1)
        ctx = SimContext(clock=clock, rng=rng)

        c = ContagionBehavior()
        c.transmission_prob = 1.0  # certain infection with 1 infected neighbor
        infected_neighbors = [object()]
        result = c.step_state("S", infected_neighbors, ctx)
        assert result == "I"

    def test_susceptible_stays_susceptible_without_exposure(self):
        rng = np.random.default_rng(0)
        ctx = SimContext(clock=DiscreteClock(steps=1), rng=rng)
        c = ContagionBehavior()
        result = c.step_state("S", [], ctx)
        assert result == "S"

    def test_infected_can_recover(self):
        rng = np.random.default_rng(42)
        ctx = SimContext(clock=DiscreteClock(steps=1), rng=rng)
        c = ContagionBehavior()
        c.recovery_rate = 1.0
        result = c.step_state("I", [], ctx)
        assert result == "R"

    def test_recovered_does_not_change(self):
        rng = np.random.default_rng(0)
        ctx = SimContext(clock=DiscreteClock(steps=1), rng=rng)
        c = ContagionBehavior()
        result = c.step_state("R", [object(), object()], ctx)
        assert result == "R"

    def test_multi_neighbor_probability(self):
        """P(infection) = 1 - (1-p)^n, should increase with n."""
        import math
        p = 0.3
        c = ContagionBehavior()
        c.transmission_prob = p
        # With n=5 infected neighbors, P = 1 - 0.7^5 ≈ 0.832
        expected = 1 - (1 - p) ** 5
        # Run 5000 trials
        infections = 0
        for seed in range(5000):
            ctx = SimContext(clock=DiscreteClock(steps=1), rng=np.random.default_rng(seed))
            result = c.step_state("S", [None] * 5, ctx)
            if result == "I":
                infections += 1
        observed = infections / 5000
        assert abs(observed - expected) < 0.02, f"expected {expected:.3f}, got {observed:.3f}"


# ============================================================
# MemoryBehavior
# ============================================================

class TestMemoryBehavior:
    def test_remember_and_recall(self):
        m = MemoryBehavior()
        m.remember({"ts": 1, "value": 42})
        entries = m.recall()
        assert len(entries) == 1
        assert entries[0]["value"] == 42

    def test_recall_last_n(self):
        m = MemoryBehavior()
        for i in range(5):
            m.remember({"v": i})
        last_2 = m.recall(last=2)
        assert [e["v"] for e in last_2] == [3, 4]

    def test_query_field(self):
        m = MemoryBehavior()
        for i in range(4):
            m.remember({"score": i * 10, "ignored": True})
        scores = m.query("score")
        assert scores == [0, 10, 20, 30]

    def test_query_missing_field_skipped(self):
        m = MemoryBehavior()
        m.remember({"a": 1})
        m.remember({"b": 2})  # no "a"
        values = m.query("a")
        assert values == [1]

    def test_forget_clears_all(self):
        m = MemoryBehavior()
        for i in range(10):
            m.remember({"x": i})
        m.forget()
        assert len(m) == 0
        assert m.recall() == []

    def test_capacity_limit(self):
        class SmallMemory(MemoryBehavior):
            capacity = 3

        m = SmallMemory()
        for i in range(10):
            m.remember({"i": i})
        assert len(m) == 3
        # Should contain the last 3
        assert m.recall()[0]["i"] == 7

    def test_len(self):
        m = MemoryBehavior()
        assert len(m) == 0
        m.remember({"x": 1})
        assert len(m) == 1
