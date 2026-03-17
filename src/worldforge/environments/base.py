"""Abstract base class for all simulation environments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    """
    Abstract base for all worldforge environments.

    An environment defines the spatial (or structural) context in which
    agents exist. Agents are registered/unregistered here as they are
    created/removed from the simulation.
    """

    def add_agent(self, agent: Any) -> None:
        """Register an agent in this environment."""

    def remove_agent(self, agent: Any) -> None:
        """Unregister an agent from this environment."""

    def agents(self) -> list:
        """Return all agents in this environment."""
        return []

    def step(self, ctx: Any) -> None:
        """Called each tick for environment-level updates (e.g., physics)."""
