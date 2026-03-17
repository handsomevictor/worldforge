"""StateMachineBehavior: declarative probabilistic FSM with dwell-time distributions."""
from __future__ import annotations

from typing import Any

import numpy as np

from worldforge.distributions.base import Distribution
from worldforge.core.exceptions import ConfigurationError


class StateMachineBehavior:
    """
    Declarative finite state machine behavior for agents.

    Define states, transitions, and dwell-time distributions as class attributes.
    Attach to an Agent via field()::

        class OrderFSM(StateMachineBehavior):
            states = ["pending", "paid", "shipped", "delivered", "cancelled"]
            initial = "pending"
            terminal = ["delivered", "cancelled"]
            transitions = {
                "pending": [
                    (0.85, "paid",       Exponential(scale=300)),
                    (0.15, "cancelled",  Exponential(scale=3600)),
                ],
                "paid": [
                    (1.00, "shipped", Exponential(scale=86400)),
                ],
            }

        class Order(Agent):
            fsm: OrderFSM = field(OrderFSM)

            def step(self, ctx):
                self.fsm.step(ctx)
                if self.fsm.is_terminal:
                    ctx.remove_agent(self)

    Transition tuples: (probability, next_state, dwell_time_distribution)
    """

    # Class-level declarations (override in subclasses)
    states: list[str] = []
    initial: str = ""
    terminal: list[str] = []
    transitions: dict[str, list[tuple]] = {}

    def __init__(self) -> None:
        self._state: str = self.initial
        self._initialized: bool = False
        self._next_state: str | None = None
        self._dwell_time: float = float("inf")
        self._time_in_state: float = 0.0
        self.agent: Any = None  # set by Agent.__init__ after field resolution

    def _enter_state(self, state: str, ctx: Any) -> None:
        """Enter `state` and sample the next transition."""
        self._state = state
        self._time_in_state = 0.0

        available = self.transitions.get(state, [])
        if not available:
            self._next_state = None
            self._dwell_time = float("inf")
            return

        # Normalize probabilities and pick which transition to take
        probs = np.array([t[0] for t in available], dtype=float)
        probs /= probs.sum()
        idx = int(ctx.rng.choice(len(available), p=probs))
        _, self._next_state, dwell_dist = available[idx]

        # Sample dwell time from the transition's distribution
        if isinstance(dwell_dist, Distribution):
            self._dwell_time = float(dwell_dist.sample(ctx.rng))
        elif isinstance(dwell_dist, (int, float)):
            self._dwell_time = float(dwell_dist)
        else:
            self._dwell_time = float("inf")

    def step(self, ctx: Any) -> None:
        """Advance the FSM by one tick. Call from Agent.step()."""
        if self.is_terminal:
            return

        # Lazy initialization: sample first transition on the very first step
        if not self._initialized:
            self._enter_state(self._state, ctx)
            self._initialized = True

        self._time_in_state += 1

        if self._time_in_state >= self._dwell_time and self._next_state is not None:
            from_state = self._state
            to_state = self._next_state
            self._enter_state(to_state, ctx)
            self.on_transition(from_state, to_state, ctx)

    def on_transition(self, from_state: str, to_state: str, ctx: Any) -> None:
        """Override to react to state transitions (e.g., emit events)."""

    @property
    def current_state(self) -> str:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state in self.terminal

    def __repr__(self) -> str:
        return f"{type(self).__name__}(state={self._state!r})"
