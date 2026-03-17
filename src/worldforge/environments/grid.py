"""GridEnvironment: 2D discrete grid with configurable topology."""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from worldforge.environments.base import Environment
from worldforge.core.exceptions import ConfigurationError


class GridEnvironment(Environment):
    """
    2D grid environment. Agents occupy (x, y) cells.

    Supports:
    - topology: 'bounded' (walls) or 'torus' (wraps around)
    - neighborhood: 'moore' (8 neighbors) or 'von_neumann' (4 neighbors)

    Example::

        env = GridEnvironment(width=50, height=50, topology="torus")
        env.place(agent, x=10, y=20)
        env.move(agent, dx=1, dy=0)
        nearby = env.neighbors(agent)
    """

    def __init__(
        self,
        width: int,
        height: int,
        topology: Literal["bounded", "torus"] = "bounded",
        neighborhood: Literal["moore", "von_neumann"] = "moore",
    ) -> None:
        if width < 1 or height < 1:
            raise ConfigurationError(f"Grid dimensions must be >= 1: {width}x{height}")
        self.width = width
        self.height = height
        self.topology = topology
        self.neighborhood = neighborhood

        # cell_map: (x, y) -> list[Agent]
        self._cell_map: dict[tuple[int, int], list[Any]] = {}
        # pos_map: agent_id -> (x, y)
        self._pos_map: dict[str, tuple[int, int]] = {}
        # agent objects
        self._agent_map: dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Environment interface
    # -------------------------------------------------------------------

    def add_agent(self, agent: Any) -> None:
        self._agent_map[agent.id] = agent

    def remove_agent(self, agent: Any) -> None:
        self._agent_map.pop(agent.id, None)
        if agent.id in self._pos_map:
            pos = self._pos_map.pop(agent.id)
            cell = self._cell_map.get(pos, [])
            try:
                cell.remove(agent)
            except ValueError:
                pass

    def agents(self) -> list:
        return list(self._agent_map.values())

    # -------------------------------------------------------------------
    # Placement
    # -------------------------------------------------------------------

    def place(self, agent: Any, x: int, y: int) -> None:
        """Place agent at grid position (x, y)."""
        x, y = self._normalize(x, y)
        # Remove from old position if any
        if agent.id in self._pos_map:
            old_pos = self._pos_map[agent.id]
            old_cell = self._cell_map.get(old_pos, [])
            try:
                old_cell.remove(agent)
            except ValueError:
                pass
        self._pos_map[agent.id] = (x, y)
        self._cell_map.setdefault((x, y), []).append(agent)

    def move(self, agent: Any, dx: int, dy: int) -> None:
        """Move agent by (dx, dy) relative to current position."""
        if agent.id not in self._pos_map:
            raise ConfigurationError(
                f"Agent {agent.id} has no position; call place() first."
            )
        cx, cy = self._pos_map[agent.id]
        self.place(agent, cx + dx, cy + dy)

    def position(self, agent: Any) -> tuple[int, int] | None:
        return self._pos_map.get(agent.id)

    # -------------------------------------------------------------------
    # Neighbor queries
    # -------------------------------------------------------------------

    def neighbors(self, agent: Any, radius: int = 1) -> list:
        """Return all agents within `radius` steps of the given agent."""
        if agent.id not in self._pos_map:
            return []
        cx, cy = self._pos_map[agent.id]
        result = []
        for nx_ in range(cx - radius, cx + radius + 1):
            for ny_ in range(cy - radius, cy + radius + 1):
                if nx_ == cx and ny_ == cy:
                    continue
                if self.neighborhood == "von_neumann":
                    if abs(nx_ - cx) + abs(ny_ - cy) > radius:
                        continue
                try:
                    nx2, ny2 = self._normalize(nx_, ny_)
                except ConfigurationError:
                    continue
                for a in self._cell_map.get((nx2, ny2), []):
                    if a is not agent:
                        result.append(a)
        return result

    def agents_at(self, x: int, y: int) -> list:
        """Return all agents at a specific cell."""
        x, y = self._normalize(x, y)
        return list(self._cell_map.get((x, y), []))

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _normalize(self, x: int, y: int) -> tuple[int, int]:
        if self.topology == "torus":
            return x % self.width, y % self.height
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ConfigurationError(
                f"Position ({x}, {y}) is out of bounds for "
                f"{self.width}x{self.height} grid."
            )
        return x, y

    def __repr__(self) -> str:
        return (
            f"GridEnvironment({self.width}x{self.height}, "
            f"topology={self.topology!r}, neighborhood={self.neighborhood!r})"
        )
