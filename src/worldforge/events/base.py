"""Base class for all simulation events."""
from __future__ import annotations

from typing import Any


class Event:
    """
    Base class for all worldforge events.

    Subclasses should be decorated with @dataclass::

        @dataclass
        class PurchaseEvent(Event):
            user_id: str
            amount: float

    ctx.emit() automatically sets the `timestamp` attribute to ctx.now.
    Users do not need to pass timestamp when constructing events.
    """

    # timestamp is set by SimContext.emit(), not by the constructor.
    # Defined as a class attribute so it always exists even before emit() is called.
    timestamp: Any = None

    # source_id can optionally be set by the emitter (agent id, "system", etc.)
    source_id: str | None = None
