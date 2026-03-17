"""SimContext — the runtime context injected into every step(), probe, and rule."""
from __future__ import annotations

from typing import Any, Callable, Type

import numpy as np

from worldforge.core.clock import Clock
from worldforge.core.exceptions import SimulationError


class SimContext:
    """
    Runtime context for a running simulation.

    Passed to agent.step(), probes, global rules, and event handlers.
    All mutation of simulation state should go through SimContext methods
    rather than directly modifying internal data structures.

    Agent additions and removals during a tick are *deferred* and applied
    at the end of the tick by _flush_pending(). This prevents iterator
    invalidation when agents remove themselves or others during step().
    """

    def __init__(
        self,
        clock: Clock,
        rng: np.random.Generator,
    ) -> None:
        self._clock = clock
        self.rng = rng

        # Agent storage
        self._agents: dict[str, Any] = {}           # id -> Agent
        self._by_type: dict[type, list[Any]] = {}   # AgentType -> [Agent]

        # Event log: list of emitted events (in emission order)
        self._event_log: list[Any] = []

        # Event handlers registered via @sim.on(EventType)
        self._event_handlers: dict[type, list[Callable]] = {}

        # Deferred mutations (processed at end of each tick)
        self._pending_removals: list[Any] = []
        self._pending_spawns: list[tuple] = []  # (agent_type, count, parent, init_fn)

        # Global rules registered via @sim.global_rule(every=...)
        # Managed by Simulation; context just stores them for the runner to call.
        self._global_rules: list[tuple[Callable, Any]] = []  # (fn, interval)

    # -------------------------------------------------------------------
    # Time
    # -------------------------------------------------------------------

    @property
    def now(self) -> Any:
        """Current simulation time (int for DiscreteClock, datetime for CalendarClock)."""
        return self._clock.now

    # -------------------------------------------------------------------
    # Event system
    # -------------------------------------------------------------------

    def emit(self, event: Any) -> None:
        """
        Emit an event: set its timestamp, log it, dispatch to handlers and agents.

        Event handlers registered via @sim.on() are called immediately.
        Agent.on_event() is called on agents that override it.
        """
        event.timestamp = self.now
        self._event_log.append(event)

        # Dispatch to global handlers (registered via @sim.on)
        for handler in self._event_handlers.get(type(event), []):
            handler(event, self)

        # Broadcast to agents that override on_event
        from worldforge.agent import Agent  # local import to avoid circular
        for agent in list(self._agents.values()):
            if type(agent).on_event is not Agent.on_event:
                agent.on_event(event, self)

    def register_event_handler(self, event_type: type, handler: Callable) -> None:
        """Register a global event handler for the given event type."""
        self._event_handlers.setdefault(event_type, []).append(handler)

    # -------------------------------------------------------------------
    # Agent queries
    # -------------------------------------------------------------------

    def agents(
        self,
        agent_type: type | None = None,
        *,
        filter: Callable | None = None,
    ) -> list:
        """
        Return a list of agents, optionally filtered by type and/or predicate.

        Results are ordered by agent ID (deterministic).
        """
        if agent_type is None:
            result = list(self._agents.values())
        else:
            result = list(self._by_type.get(agent_type, []))
        if filter is not None:
            result = [a for a in result if filter(a)]
        return result

    def agent_count(self, agent_type: type | None = None) -> int:
        if agent_type is None:
            return len(self._agents)
        return len(self._by_type.get(agent_type, []))

    def agent_mean(self, agent_type: type, field_name: str) -> float:
        agents = self._by_type.get(agent_type, [])
        if not agents:
            return 0.0
        return sum(getattr(a, field_name) for a in agents) / len(agents)

    def agent_percentile(self, agent_type: type, field_name: str, q: float) -> float:
        agents = self._by_type.get(agent_type, [])
        if not agents:
            return 0.0
        values = sorted(getattr(a, field_name) for a in agents)
        idx = int(q * len(values))
        return values[min(idx, len(values) - 1)]

    def get_agent(self, agent_id: str) -> Any | None:
        return self._agents.get(agent_id)

    # -------------------------------------------------------------------
    # Event queries
    # -------------------------------------------------------------------

    def event_sum(self, event_type: type, field_name: str, last=None) -> float:
        """Sum `field_name` across all events of `event_type` (optionally within last N steps)."""
        return sum(
            getattr(e, field_name, 0.0)
            for e in self._filter_events(event_type, last)
        )

    def event_count(self, event_type: type, last=None) -> int:
        """Count events of `event_type` (optionally within last N steps)."""
        return len(self._filter_events(event_type, last))

    def event_rate(self, event_type: type, window=None) -> float:
        """Events per step within the given window."""
        count = self.event_count(event_type, last=window)
        if window is None or not isinstance(window, int):
            denom = max(int(self.now), 1)
        else:
            denom = window
        return count / denom

    def _filter_events(self, event_type: type, last=None) -> list:
        relevant = [e for e in self._event_log if isinstance(e, event_type)]
        if last is None:
            return relevant
        if isinstance(last, int):
            # DiscreteClock: last N steps
            cutoff = self.now - last
            return [e for e in relevant if e.timestamp is not None and e.timestamp >= cutoff]
        if isinstance(last, str):
            # CalendarClock: duration string like "1 day", "2 hours"
            try:
                from worldforge.time.calendar import parse_duration
                delta = parse_duration(last)
                cutoff = self.now - delta
                return [e for e in relevant if e.timestamp is not None and e.timestamp >= cutoff]
            except Exception:
                return relevant
        # timedelta passthrough
        from datetime import timedelta
        if isinstance(last, timedelta):
            cutoff = self.now - last
            return [e for e in relevant if e.timestamp is not None and e.timestamp >= cutoff]
        return relevant  # unsupported last type: return all

    # -------------------------------------------------------------------
    # Agent mutation (deferred)
    # -------------------------------------------------------------------

    def remove_agent(self, agent: Any) -> None:
        """Queue an agent for removal at the end of the current tick."""
        if agent not in self._pending_removals:
            self._pending_removals.append(agent)

    def spawn(
        self,
        agent_type: type,
        count: int = 1,
        parent: Any = None,
        init: Callable | None = None,
    ) -> None:
        """Queue `count` new agents of `agent_type` to be added after the current tick."""
        self._pending_spawns.append((agent_type, count, parent, init))

    # -------------------------------------------------------------------
    # Internal: agent registration (used by Simulation/runner)
    # -------------------------------------------------------------------

    def _register_agent(self, agent: Any) -> None:
        self._agents[agent.id] = agent
        self._by_type.setdefault(type(agent), []).append(agent)

    def _unregister_agent(self, agent: Any) -> None:
        self._agents.pop(agent.id, None)
        type_list = self._by_type.get(type(agent), [])
        try:
            type_list.remove(agent)
        except ValueError:
            pass

    # -------------------------------------------------------------------
    # Internal: tick execution (called by runner)
    # -------------------------------------------------------------------

    def _run_tick(self) -> None:
        """
        Execute one simulation tick:
        1. Call step() on every agent (in stable order).
        2. Flush deferred removals and spawns.
        """
        for agent in list(self._agents.values()):
            agent._ctx = self
            try:
                agent.step(self)
            finally:
                agent._ctx = None
        self._flush_pending()

    def _flush_pending(self) -> None:
        """Apply deferred agent removals and spawns."""
        for agent in self._pending_removals:
            agent.on_die(self)
            self._unregister_agent(agent)
        self._pending_removals.clear()

        for agent_type, count, parent, init_fn in self._pending_spawns:
            for _ in range(count):
                agent = agent_type(_rng=self.rng)
                if init_fn is not None:
                    init_fn(agent)
                self._register_agent(agent)
                agent.on_born(self)
        self._pending_spawns.clear()
