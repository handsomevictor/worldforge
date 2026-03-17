"""Tests for Agent base class and field() declaration system (Step 07)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from worldforge.agent import Agent, field, _reset_id_counter
from worldforge.distributions import Normal, Categorical, Uniform, ConditionalDistribution
from worldforge.core.exceptions import AgentError


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset the global ID counter before each test for predictable IDs."""
    _reset_id_counter(1)
    yield


class TestFieldDeclaration:
    def test_constant_field(self, rng):
        class A(Agent):
            x: int = field(42)

        a = A(_rng=rng)
        assert a.x == 42

    def test_distribution_field(self, rng):
        class A(Agent):
            value: float = field(Uniform(0, 1))

        samples = [A(_rng=np.random.default_rng(i)).value for i in range(100)]
        assert all(0 <= v <= 1 for v in samples)
        assert len(set(samples)) > 1  # not all identical

    def test_lambda_field_references_id(self, rng):
        class A(Agent):
            name: str = field(lambda a: f"agent_{a.id}")

        a = A(_rng=rng)
        assert a.name == f"agent_{a.id}"

    def test_lambda_field_references_earlier_field(self, rng):
        class A(Agent):
            tier: str = field("pro")
            label: str = field(lambda a: f"{a.tier}_user")

        a = A(_rng=rng)
        assert a.label == "pro_user"

    def test_conditional_distribution_field(self, rng):
        class A(Agent):
            tier: str = field("enterprise")
            bonus: float = field(ConditionalDistribution(
                condition=lambda a: a.tier,
                mapping={
                    "free":       Uniform(0, 1),
                    "enterprise": Uniform(1000, 2000),
                },
            ))

        a = A(_rng=rng)
        assert 1000 <= a.bonus <= 2000

    def test_override_field_at_construction(self, rng):
        class A(Agent):
            x: float = field(Normal(0, 1))

        a = A(_rng=rng, x=999.0)
        assert a.x == 999.0

    def test_field_not_a_class_attribute(self):
        """After metaclass processing, FieldSpec should not be accessible on the class."""
        class A(Agent):
            x: int = field(10)

        assert "x" not in A.__dict__

    def test_fields_not_shared_between_instances(self, rng):
        class A(Agent):
            items: list = field(lambda _: [])

        a1 = A(_rng=rng)
        a2 = A(_rng=rng)
        a1.items.append(1)
        assert a2.items == []

    def test_multiple_fields(self, rng):
        class User(Agent):
            balance: float = field(Normal(5000, 100, clip=(0, None)))
            tier: str = field(Categorical(["free", "pro"], [0.5, 0.5]))
            age_days: int = field(0)

        u = User(_rng=rng)
        assert u.balance >= 0
        assert u.tier in ("free", "pro")
        assert u.age_days == 0


class TestFieldInheritance:
    def test_child_inherits_parent_fields(self, rng):
        class Base(Agent):
            x: int = field(1)

        class Child(Base):
            y: int = field(2)

        c = Child(_rng=rng)
        assert c.x == 1
        assert c.y == 2

    def test_child_overrides_parent_field(self, rng):
        class Base(Agent):
            x: int = field(1)

        class Child(Base):
            x: int = field(99)

        c = Child(_rng=rng)
        assert c.x == 99

    def test_parent_unaffected_by_child_fields(self, rng):
        class Base(Agent):
            x: int = field(1)

        class Child(Base):
            y: int = field(2)

        b = Base(_rng=rng)
        assert not hasattr(b, "y")


class TestAgentID:
    def test_unique_ids(self, rng):
        ids = {Agent(_rng=rng).id for _ in range(100)}
        assert len(ids) == 100

    def test_ids_are_strings(self, rng):
        a = Agent(_rng=rng)
        assert isinstance(a.id, str)

    def test_sequential_ids_after_reset(self):
        _reset_id_counter(1)
        rng = np.random.default_rng(0)
        a1 = Agent(_rng=rng)
        a2 = Agent(_rng=rng)
        assert int(a2.id) == int(a1.id) + 1


class TestLifecycleHooks:
    def test_on_born_called(self):
        log = []

        class A(Agent):
            def on_born(self, ctx):
                log.append("born")

        A().on_born(ctx=None)
        assert log == ["born"]

    def test_on_die_called(self):
        log = []

        class A(Agent):
            def on_die(self, ctx):
                log.append("died")

        A().on_die(ctx=None)
        assert log == ["died"]

    def test_step_default_is_noop(self, rng):
        a = Agent(_rng=rng)
        a.step(ctx=None)  # should not raise


class TestEmit:
    def test_emit_outside_step_raises(self, rng):
        @dataclass
        class DummyEvent:
            value: int

        a = Agent(_rng=rng)
        with pytest.raises(AgentError, match="outside of step"):
            a.emit(DummyEvent(value=1))

    def test_emit_during_step_works(self, rng):
        from worldforge.events.base import Event
        from worldforge.core.context import SimContext
        from worldforge.core.clock import DiscreteClock

        @dataclass
        class DummyEvent(Event):
            value: int

        class A(Agent):
            def step(self, ctx):
                self.emit(DummyEvent(value=42))

        clock = DiscreteClock(steps=1)
        ctx = SimContext(clock=clock, rng=rng)
        a = A(_rng=rng)
        ctx._register_agent(a)
        ctx._run_tick()
        assert any(
            isinstance(e, DummyEvent) and e.value == 42
            for e in ctx._event_log
        )
