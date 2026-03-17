"""LifecycleBehavior: age-based life events (birth cohort, aging, death)."""
from __future__ import annotations

from typing import Any

from worldforge.distributions.base import Distribution


class LifecycleBehavior:
    """
    Tracks an agent's age and manages lifecycle events.

    Attach to an Agent via field()::

        class Person(Agent):
            lifecycle: LifecycleBehavior = field(
                lambda a: LifecycleBehavior(lifespan=Normal(70, 10))
            )

    Override `on_lifecycle_event(event, ctx)` to react to age milestones.
    """

    def __init__(
        self,
        lifespan: Distribution | float | None = None,
        birth_age: int = 0,
    ) -> None:
        self.age: int = birth_age
        self._lifespan: float | None = None
        self._lifespan_dist = lifespan
        self.agent: Any = None

    def _resolve_lifespan(self, ctx: Any) -> None:
        if self._lifespan is not None:
            return
        if self._lifespan_dist is None:
            self._lifespan = float("inf")
        elif isinstance(self._lifespan_dist, Distribution):
            self._lifespan = float(self._lifespan_dist.sample(ctx.rng))
        else:
            self._lifespan = float(self._lifespan_dist)

    def step(self, ctx: Any) -> None:
        """Advance age by one tick. Remove agent if lifespan exceeded."""
        self._resolve_lifespan(ctx)
        self.age += 1
        self.on_lifecycle_event("age", ctx)
        if self._lifespan is not None and self.age >= self._lifespan:
            self.on_lifecycle_event("death", ctx)
            if self.agent is not None:
                ctx.remove_agent(self.agent)

    def on_lifecycle_event(self, event: str, ctx: Any) -> None:
        """Override to react to lifecycle events ('age', 'death')."""

    @property
    def is_alive(self) -> bool:
        return self._lifespan is None or self.age < self._lifespan
