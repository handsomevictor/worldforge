"""social_network_world: opinion dynamics on a social graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.distributions import Uniform, Normal
from worldforge.events.base import Event
from worldforge.probes import AggregatorProbe, TimeSeriesProbe


@dataclass
class OpinionShiftEvent(Event):
    user_id: str
    old_opinion: float
    new_opinion: float


class SocialUser(Agent):
    opinion: float = field(Uniform(-1.0, 1.0))
    stubbornness: float = field(Uniform(0.1, 0.9))

    def step(self, ctx: Any) -> None:
        # Get neighbors from network environment if available
        env = getattr(ctx, "environment", None)
        if env is None:
            return
        neighbors = env.neighbors(self.id) if hasattr(env, "neighbors") else []
        if not neighbors:
            return

        neighbor_mean = sum(n.opinion for n in neighbors) / len(neighbors)
        influence = (1 - self.stubbornness) * 0.1
        old_op = self.opinion
        self.opinion += influence * (neighbor_mean - self.opinion)
        self.opinion = max(-1.0, min(1.0, self.opinion))
        if abs(self.opinion - old_op) > 0.05:
            ctx.emit(OpinionShiftEvent(
                user_id=self.id,
                old_opinion=old_op,
                new_opinion=self.opinion,
            ))


def social_network_world(
    n_users: int = 1000,
    duration_steps: int = 100,
    network_type: str = "small_world",
    seed: int = 42,
) -> Simulation:
    """
    Build an opinion dynamics simulation on a social network.

    network_type: 'small_world', 'scale_free', or 'erdos_renyi'
    """
    from worldforge.core.clock import DiscreteClock

    sim = Simulation(
        name="social_network_world",
        seed=seed,
        clock=DiscreteClock(steps=duration_steps),
    )
    sim.add_agents(SocialUser, count=n_users)

    sim.add_probe(TimeSeriesProbe(
        series={
            "mean_opinion": lambda ctx: ctx.agent_mean(SocialUser, "opinion"),
        },
        every=1,
        name="opinion_timeseries",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "polarization": lambda ctx: (
                sum(abs(u.opinion) for u in ctx.agents(SocialUser)) /
                max(1, ctx.agent_count(SocialUser))
            ),
        },
        every=5,
        name="polarization",
    ))

    return sim
