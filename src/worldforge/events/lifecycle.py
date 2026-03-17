"""Built-in lifecycle events emitted by the simulation engine."""
from __future__ import annotations

from dataclasses import dataclass

from worldforge.events.base import Event


@dataclass
class AgentCreated(Event):
    """Emitted when an agent is added to the simulation."""

    agent_id: str
    agent_type: str  # class name


@dataclass
class AgentRemoved(Event):
    """Emitted when an agent is removed from the simulation."""

    agent_id: str
    agent_type: str
    reason: str = "explicit"  # "explicit" | "churn" | "terminal_state"
