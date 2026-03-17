"""Tests for EventQueue (Step 08)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from worldforge.core.event_queue import EventQueue
from worldforge.events.base import Event


@dataclass
class FakeEvent(Event):
    name: str


class TestEventQueueOrdering:
    def test_pop_returns_earliest_first(self):
        q = EventQueue()
        e1, e2, e3 = FakeEvent("a"), FakeEvent("b"), FakeEvent("c")
        q.schedule(e3, at=30)
        q.schedule(e1, at=10)
        q.schedule(e2, at=20)

        ev, t = q.pop()
        assert t == 10 and ev.name == "a"
        ev, t = q.pop()
        assert t == 20 and ev.name == "b"
        ev, t = q.pop()
        assert t == 30 and ev.name == "c"

    def test_fifo_within_same_timestamp(self):
        """Events at the same time must come out in insertion order."""
        q = EventQueue()
        names = ["first", "second", "third"]
        for name in names:
            q.schedule(FakeEvent(name), at=5)

        out = []
        while not q.is_empty():
            ev, _ = q.pop()
            out.append(ev.name)

        assert out == names

    def test_single_element(self):
        q = EventQueue()
        e = FakeEvent("only")
        q.schedule(e, at=1)
        ev, t = q.pop()
        assert ev is e and t == 1

    def test_pop_empty_raises(self):
        q = EventQueue()
        with pytest.raises(IndexError):
            q.pop()


class TestEventQueueState:
    def test_len_tracks_size(self):
        q = EventQueue()
        assert len(q) == 0
        q.schedule(FakeEvent("x"), at=1)
        assert len(q) == 1
        q.schedule(FakeEvent("y"), at=2)
        assert len(q) == 2
        q.pop()
        assert len(q) == 1

    def test_is_empty(self):
        q = EventQueue()
        assert q.is_empty()
        q.schedule(FakeEvent("x"), at=0)
        assert not q.is_empty()
        q.pop()
        assert q.is_empty()

    def test_peek_time(self):
        q = EventQueue()
        assert q.peek_time() is None
        q.schedule(FakeEvent("a"), at=10)
        q.schedule(FakeEvent("b"), at=5)
        assert q.peek_time() == 5

    def test_peek_does_not_consume(self):
        q = EventQueue()
        q.schedule(FakeEvent("x"), at=1)
        q.peek_time()
        assert len(q) == 1

    def test_many_events_correct_order(self):
        import random
        rng = random.Random(42)
        q = EventQueue()
        times = [rng.randint(0, 100) for _ in range(200)]
        for t in times:
            q.schedule(FakeEvent(str(t)), at=t)

        prev_t = -1
        while not q.is_empty():
            _, t = q.pop()
            assert t >= prev_t
            prev_t = t
