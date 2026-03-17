"""Agent base class and field() declaration system."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from worldforge.distributions.base import Distribution
from worldforge.distributions.conditional import ConditionalDistribution
from worldforge.core.exceptions import AgentError

# ---------------------------------------------------------------------------
# Global ID counter — sequential within a process run.
# The simulation resets this at startup for reproducibility.
# ---------------------------------------------------------------------------
_global_id_counter = itertools.count(1)


def _next_id() -> str:
    return str(next(_global_id_counter))


def _reset_id_counter(start: int = 1) -> None:
    """Reset global agent ID counter. Intended for use in tests only."""
    global _global_id_counter
    _global_id_counter = itertools.count(start)


# ---------------------------------------------------------------------------
# FieldSpec — wraps a field initializer
# ---------------------------------------------------------------------------

class FieldSpec:
    """
    Stores field initialization information for Agent subclasses.
    Created by field() and consumed by AgentMeta / Agent.__init__.
    """

    def __init__(self, initializer: Any) -> None:
        self.initializer = initializer

    def resolve(self, agent: "Agent", rng: np.random.Generator) -> Any:
        """
        Evaluate the field value for `agent` using `rng`.

        Evaluation order:
        1. ConditionalDistribution → sample with agent as context
        2. Any other Distribution  → sample
        3. Any other callable      → call with agent (lambda fields)
        4. Constant                → return as-is
        """
        init = self.initializer
        if isinstance(init, ConditionalDistribution):
            return init.sample(rng, context=agent)
        if isinstance(init, Distribution):
            return init.sample(rng)
        if callable(init):
            return init(agent)
        return init


def field(initializer: Any) -> "FieldSpec":
    """
    Declare an Agent field with the given initializer.

    Accepts:
    - A constant value:           field(0)
    - A Distribution instance:   field(Normal(5000, 1000))
    - A callable (lambda):        field(lambda agent: f"user_{agent.id}@example.com")
    - A ConditionalDistribution:  field(ConditionalDistribution(...))
    """
    return FieldSpec(initializer)


# ---------------------------------------------------------------------------
# AgentMeta — collects FieldSpec declarations from class body
# ---------------------------------------------------------------------------

class AgentMeta(type):
    """
    Metaclass that scans each Agent subclass for FieldSpec attributes,
    collects them into cls._fields (preserving declaration order), and
    removes them from the class namespace so they don't shadow instance attrs.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple,
        namespace: dict,
        **kwargs: Any,
    ) -> "AgentMeta":
        # Inherit fields from base classes (MRO order: child overrides parent)
        inherited: dict[str, FieldSpec] = {}
        for base in reversed(bases):
            if hasattr(base, "_fields"):
                inherited.update(base._fields)

        # Collect FieldSpecs from this class and remove from namespace
        own_fields: dict[str, FieldSpec] = {}
        to_delete: list[str] = []
        for attr_name, value in namespace.items():
            if isinstance(value, FieldSpec):
                own_fields[attr_name] = value
                to_delete.append(attr_name)
        for attr_name in to_delete:
            del namespace[attr_name]

        # Merge: parent fields first, then child fields (child wins on collision)
        namespace["_fields"] = {**inherited, **own_fields}
        return super().__new__(mcs, name, bases, namespace, **kwargs)


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------

class Agent(metaclass=AgentMeta):
    """
    Base class for all worldforge agents.

    Fields are declared using field()::

        class User(Agent):
            balance: float = field(Normal(5000, 1000))
            tier: str = field(Categorical(["free", "pro"], [0.7, 0.3]))
            email: str = field(lambda a: f"{a.id}@example.com")

    Override step() to define per-tick behavior. Use lifecycle hooks
    (on_born, on_die, on_event) as needed.
    """

    _fields: dict[str, FieldSpec] = {}

    def __init__(
        self,
        _rng: np.random.Generator | None = None,
        **overrides: Any,
    ) -> None:
        # ID is assigned immediately so lambda fields can reference it
        self.id: str = _next_id()

        # _ctx is injected by the runner before each step() call and cleared after.
        # emit() uses it. Do NOT store ctx beyond a single step.
        self._ctx = None

        rng = _rng if _rng is not None else np.random.default_rng()

        # Evaluate fields in declaration order (parent fields first).
        # This ensures that fields referenced by lambda/conditional fields
        # are already set when their dependents are evaluated.
        for name, spec in type(self)._fields.items():
            if name in overrides:
                object.__setattr__(self, name, overrides[name])
            else:
                object.__setattr__(self, name, spec.resolve(self, rng))

    # --- Lifecycle hooks (override in subclasses) ---

    def on_born(self, ctx) -> None:
        """Called once when the agent enters the simulation."""

    def step(self, ctx) -> None:
        """Called every simulation tick. Define agent behavior here."""

    def on_die(self, ctx) -> None:
        """Called once when the agent is removed from the simulation."""

    def on_event(self, event, ctx) -> None:
        """Called when any event is emitted (broadcast model)."""

    # --- Utilities ---

    def emit(self, event) -> None:
        """
        Emit an event to the simulation bus.
        Only valid when called from within step().
        """
        if self._ctx is None:
            raise AgentError(
                f"{type(self).__name__}.emit() called outside of step(). "
                "Only call self.emit() from within step(), or use ctx.emit() directly."
            )
        self._ctx.emit(event)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"
