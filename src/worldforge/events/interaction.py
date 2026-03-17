"""AgentInteraction event: represents a direct agent-to-agent interaction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge.events.base import Event


@dataclass
class AgentInteraction(Event):
    """
    Records a direct interaction between two agents.

    Fields
    ------
    initiator_id:   ID of the agent that initiated the interaction.
    target_id:      ID of the agent that received the interaction.
    interaction_type: String tag (e.g., 'trade', 'message', 'infection').
    payload:        Optional dict carrying interaction-specific data.
    """
    initiator_id: str
    target_id: str
    interaction_type: str
    payload: dict | None = None
