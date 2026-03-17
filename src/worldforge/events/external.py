"""External shocks: one-time world-altering events scheduled at a specific time."""
from __future__ import annotations

from typing import Any, Callable


class ExternalShock:
    """
    A scheduled world-altering event that fires at a specific simulation time.

    Usage::

        sim.add_shock(ExternalShock(
            at="2024-06-15",
            description="Competitor launched free tier",
            effect=lambda ctx: [
                setattr(u, "churn_risk", min(u.churn_risk * 3, 1.0))
                for u in ctx.agents(User)
            ],
        ))
    """

    def __init__(
        self,
        at: Any,
        effect: Callable,
        description: str = "",
    ) -> None:
        self.at = at
        self.effect = effect
        self.description = description
        self._fired = False

    def fire(self, ctx) -> None:
        """Execute the shock effect. Called once by the simulation runner."""
        if not self._fired:
            self.effect(ctx)
            self._fired = True

    def __repr__(self) -> str:
        return f"ExternalShock(at={self.at!r}, description={self.description!r})"
