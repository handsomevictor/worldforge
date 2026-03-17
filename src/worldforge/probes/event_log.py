"""EventLogProbe: records all emitted events of specified types."""
from __future__ import annotations

from typing import Any

from worldforge.probes.base import Probe


def _event_to_dict(event: Any) -> dict:
    """Convert an event object to a plain dict, skipping private attributes."""
    result = {}
    for key, val in vars(event).items():
        if not key.startswith("_"):
            result[key] = val
    return result


class EventLogProbe(Probe):
    """
    Records all events of the specified types emitted during the simulation.

    Example::

        sim.add_probe(EventLogProbe(
            events=[PurchaseEvent, ChurnEvent],
            name="event_log",
        ))
    """

    def __init__(
        self,
        events: list[type],
        name: str = "event_log",
    ) -> None:
        super().__init__(every=1, name=name)
        self._event_types = tuple(events)
        self._records: list[dict] = []
        self._last_idx: int = 0  # index into ctx._event_log for incremental collection

    def collect(self, ctx: Any) -> None:
        for event in ctx._event_log[self._last_idx:]:
            if isinstance(event, self._event_types):
                self._records.append(_event_to_dict(event))
        self._last_idx = len(ctx._event_log)

    def finalize(self) -> list:
        return list(self._records)
