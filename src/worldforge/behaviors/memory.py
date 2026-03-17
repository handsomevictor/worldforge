"""MemoryBehavior: bounded episodic memory for agents."""
from __future__ import annotations

from collections import deque
from typing import Any


class MemoryBehavior:
    """
    A bounded circular buffer of memory entries (dicts).

    Agents can record observations and later query them.

    Example::

        class UserMemory(MemoryBehavior):
            capacity = 30  # remember last 30 events

        class User(Agent):
            memory: UserMemory = field(UserMemory)

            def step(self, ctx):
                # record something
                self.memory.remember({"ts": ctx.now, "price": current_price})

                # query: average price in memory
                prices = self.memory.query("price")
                avg = sum(prices) / len(prices) if prices else 0.0
    """

    capacity: int = 100

    def __init__(self) -> None:
        self.agent: Any = None
        self._buffer: deque[dict] = deque(maxlen=self.capacity)

    def remember(self, entry: dict) -> None:
        """Store a memory entry."""
        self._buffer.append(entry)

    def recall(self, last: int | None = None) -> list[dict]:
        """Return stored memories, optionally limited to the last N entries."""
        entries = list(self._buffer)
        if last is not None:
            return entries[-last:]
        return entries

    def query(self, field_name: str, last: int | None = None) -> list:
        """Return values of `field_name` from stored memories."""
        return [
            entry[field_name]
            for entry in self.recall(last)
            if field_name in entry
        ]

    def forget(self) -> None:
        """Clear all memories."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"MemoryBehavior(len={len(self._buffer)}, capacity={self.capacity})"
