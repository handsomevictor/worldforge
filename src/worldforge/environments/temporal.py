"""TemporalEnvironment: no spatial structure, agents exist only in time."""
from __future__ import annotations

from typing import Any

from worldforge.environments.base import Environment


class TemporalEnvironment(Environment):
    """
    Default environment with no spatial structure.

    Agents are tracked by id, but there is no neighborhood, position,
    or topology. This is the default environment for Simulation.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}

    def add_agent(self, agent: Any) -> None:
        self._agents[agent.id] = agent

    def remove_agent(self, agent: Any) -> None:
        self._agents.pop(agent.id, None)

    def agents(self) -> list:
        return list(self._agents.values())

    def step(self, ctx: Any) -> None:
        pass
