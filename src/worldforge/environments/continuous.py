"""ContinuousSpace: 2D continuous coordinate space."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldforge.environments.base import Environment
from worldforge.core.exceptions import ConfigurationError


class ContinuousSpace(Environment):
    """
    2D continuous Euclidean space for agent positioning.

    Agents have floating-point (x, y) coordinates. Supports
    bounded (walls) and toroidal topology.

    Example::

        env = ContinuousSpace(width=100.0, height=100.0, topology="torus")
        env.place(agent, x=10.5, y=20.3)
        env.move(agent, dx=1.0, dy=0.5)
        nearby = env.agents_near(agent, radius=5.0)
    """

    def __init__(
        self,
        width: float,
        height: float,
        topology: str = "bounded",
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.topology = topology
        self._pos_map: dict[str, tuple[float, float]] = {}  # agent_id -> (x, y)
        self._agent_map: dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Environment interface
    # -------------------------------------------------------------------

    def add_agent(self, agent: Any) -> None:
        self._agent_map[agent.id] = agent

    def remove_agent(self, agent: Any) -> None:
        self._agent_map.pop(agent.id, None)
        self._pos_map.pop(agent.id, None)

    def agents(self) -> list:
        return list(self._agent_map.values())

    # -------------------------------------------------------------------
    # Placement
    # -------------------------------------------------------------------

    def place(self, agent: Any, x: float, y: float) -> None:
        """Place agent at continuous position (x, y)."""
        self._pos_map[agent.id] = self._clamp(x, y)

    def move(self, agent: Any, dx: float, dy: float) -> None:
        """Move agent by (dx, dy)."""
        if agent.id not in self._pos_map:
            raise ConfigurationError(
                f"Agent {agent.id} has no position; call place() first."
            )
        cx, cy = self._pos_map[agent.id]
        self._pos_map[agent.id] = self._clamp(cx + dx, cy + dy)

    def position(self, agent: Any) -> tuple[float, float] | None:
        return self._pos_map.get(agent.id)

    # -------------------------------------------------------------------
    # Spatial queries
    # -------------------------------------------------------------------

    def agents_near(
        self,
        agent: Any,
        radius: float,
        agent_type: type | None = None,
    ) -> list:
        """Return agents within Euclidean radius of the given agent."""
        if agent.id not in self._pos_map:
            return []
        ax, ay = self._pos_map[agent.id]
        result = []
        for other in self._agent_map.values():
            if other is agent:
                continue
            if agent_type is not None and not isinstance(other, agent_type):
                continue
            if other.id not in self._pos_map:
                continue
            bx, by = self._pos_map[other.id]
            if self._distance(ax, ay, bx, by) <= radius:
                result.append(other)
        return result

    def distance(self, a: Any, b: Any) -> float:
        pa = self._pos_map.get(a.id)
        pb = self._pos_map.get(b.id)
        if pa is None or pb is None:
            raise ConfigurationError("Both agents must have positions.")
        return self._distance(pa[0], pa[1], pb[0], pb[1])

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        if self.topology == "torus":
            return x % self.width, y % self.height
        return (
            max(0.0, min(x, self.width)),
            max(0.0, min(y, self.height)),
        )

    def _distance(self, ax: float, ay: float, bx: float, by: float) -> float:
        if self.topology == "torus":
            dx = abs(ax - bx)
            dy = abs(ay - by)
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
            return math.sqrt(dx * dx + dy * dy)
        dx = ax - bx
        dy = ay - by
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self) -> str:
        return (
            f"ContinuousSpace({self.width}x{self.height}, "
            f"topology={self.topology!r})"
        )
