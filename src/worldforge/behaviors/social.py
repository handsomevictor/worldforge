"""SocialBehavior: influence and opinion dynamics between agents."""
from __future__ import annotations

from typing import Any, Callable


class SocialBehavior:
    """
    Models social influence: an agent's opinion/state drifts toward
    the mean of its neighbors' opinions, weighted by edge strength.

    Example::

        class UserSocial(SocialBehavior):
            opinion_field = "sentiment"
            influence_rate = 0.1

        class User(Agent):
            sentiment: float = field(Uniform(-1, 1))
            social: UserSocial = field(UserSocial)

            def step(self, ctx):
                neighbors = ctx.environment.neighbors(self.id)
                self.social.influence(neighbors, ctx)
    """

    # Override in subclass
    opinion_field: str = "opinion"
    influence_rate: float = 0.1
    conformity_bias: float = 0.0     # 0.0 = neutral, >0 pulls toward positive end

    def __init__(self) -> None:
        self.agent: Any = None

    def influence(self, neighbors: list, ctx: Any) -> None:
        """Update agent's opinion field based on neighbor opinions."""
        if not neighbors:
            return
        neighbor_mean = sum(
            getattr(n, self.opinion_field, 0.0) for n in neighbors
        ) / len(neighbors)
        current = getattr(self.agent, self.opinion_field, 0.0)
        delta = self.influence_rate * (neighbor_mean - current)
        if self.conformity_bias != 0.0:
            delta += self.conformity_bias * (0.0 - current)
        setattr(self.agent, self.opinion_field, current + delta)

    def step(self, ctx: Any, neighbors: list | None = None) -> None:
        """
        Convenience method: compute influence if neighbors are provided.
        If neighbors is None, does nothing (caller must provide neighbors).
        """
        if neighbors is not None:
            self.influence(neighbors, ctx)


class ContagionBehavior:
    """
    SIR-style contagion: susceptible → infected → recovered.

    The agent maintains a `state` attribute ('S', 'I', or 'R').
    Call `expose()` each tick with infected neighbors.

    Example::

        class PersonContagion(ContagionBehavior):
            transmission_prob = 0.3
            recovery_rate = 0.05

        class Person(Agent):
            contagion: PersonContagion = field(PersonContagion)
            state: str = field("S")

            def step(self, ctx):
                neighbors = ctx.environment.neighbors(self.id)
                infected_neighbors = [n for n in neighbors if n.state == "I"]
                self.state = self.contagion.step_state(self.state, infected_neighbors, ctx)
    """

    transmission_prob: float = 0.3
    recovery_rate: float = 0.05

    def __init__(self) -> None:
        self.agent: Any = None

    def step_state(self, state: str, infected_neighbors: list, ctx: Any) -> str:
        """Return next state given current state and infected neighbors."""
        if state == "S" and infected_neighbors:
            # Probability of getting infected = 1 - (1-p)^n
            p_infect = 1 - (1 - self.transmission_prob) ** len(infected_neighbors)
            if ctx.rng.random() < p_infect:
                return "I"
        elif state == "I":
            if ctx.rng.random() < self.recovery_rate:
                return "R"
        return state
