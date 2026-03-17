"""Priority-queue event scheduler for discrete-event simulation."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _QueueEntry:
    """
    Internal heap entry.
    Ordered by (time, seq) so that identical timestamps are broken by
    insertion order (FIFO within the same timestamp).
    """

    time: Any  # comparable simulation time (int or datetime)
    seq: int  # monotonically increasing insertion counter
    event: Any = field(compare=False)  # the actual event object


class EventQueue:
    """
    Min-heap event queue: pop() always returns the earliest-scheduled event.

    Supports any comparable time type (int for DiscreteClock, datetime for
    CalendarClock). Events with identical timestamps are returned in FIFO order.
    """

    def __init__(self) -> None:
        self._heap: list[_QueueEntry] = []
        self._seq: int = 0

    def schedule(self, event: Any, at: Any) -> None:
        """Schedule `event` to fire at simulation time `at`."""
        entry = _QueueEntry(time=at, seq=self._seq, event=event)
        self._seq += 1
        heapq.heappush(self._heap, entry)

    def pop(self) -> tuple[Any, Any]:
        """
        Remove and return (event, time) for the earliest-scheduled event.
        Raises IndexError if the queue is empty.
        """
        if not self._heap:
            raise IndexError("EventQueue is empty")
        entry = heapq.heappop(self._heap)
        return entry.event, entry.time

    def peek_time(self) -> Any | None:
        """Return the time of the next event without removing it. None if empty."""
        if not self._heap:
            return None
        return self._heap[0].time

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __repr__(self) -> str:
        return f"EventQueue(size={len(self._heap)})"
