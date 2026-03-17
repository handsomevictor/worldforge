"""saas_world: SaaS product user lifecycle simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.distributions import Normal, Categorical, Poisson
from worldforge.events.base import Event
from worldforge.probes import AggregatorProbe, EventLogProbe


@dataclass
class SubscriptionEvent(Event):
    user_id: str
    plan: str
    mrr: float


@dataclass
class UpgradeEvent(Event):
    user_id: str
    from_plan: str
    to_plan: str


@dataclass
class CancellationEvent(Event):
    user_id: str
    plan: str
    tenure_days: int


_MRR = {"free": 0.0, "starter": 29.0, "pro": 99.0, "enterprise": 499.0}


class SaaSUser(Agent):
    plan: str = field(Categorical(
        choices=["free", "starter", "pro", "enterprise"],
        weights=[0.60, 0.25, 0.12, 0.03],
    ))
    tenure_days: int = field(0)
    feature_usage: float = field(Normal(mu=0.5, sigma=0.2, clip=(0, 1)))
    nps_score: int = field(Normal(mu=7, sigma=2, clip=(0, 10)))

    def step(self, ctx: Any) -> None:
        self.tenure_days += 1

        # Usage drift
        delta = float(Normal(mu=0.0, sigma=0.01).sample(ctx.rng))
        self.feature_usage = max(0.0, min(1.0, self.feature_usage + delta))

        # Upgrade
        if self.plan in ("free", "starter") and self.feature_usage > 0.7:
            if ctx.rng.random() < 0.005:
                new_plan = "pro" if self.plan == "starter" else "starter"
                ctx.emit(UpgradeEvent(
                    user_id=self.id,
                    from_plan=self.plan,
                    to_plan=new_plan,
                ))
                self.plan = new_plan

        # Churn
        churn_prob = 0.001 + (1 - self.feature_usage) * 0.002
        if self.plan == "free":
            churn_prob += 0.005
        if ctx.rng.random() < churn_prob:
            ctx.emit(CancellationEvent(
                user_id=self.id,
                plan=self.plan,
                tenure_days=self.tenure_days,
            ))
            ctx.remove_agent(self)


def saas_world(
    n_users: int = 2000,
    duration_days: int = 365,
    seed: int = 42,
) -> Simulation:
    """Build a SaaS product user lifecycle simulation."""
    from worldforge.core.clock import DiscreteClock

    sim = Simulation(
        name="saas_world",
        seed=seed,
        clock=DiscreteClock(steps=duration_days),
    )
    sim.add_agents(SaaSUser, count=n_users)

    sim.add_probe(EventLogProbe(
        events=[UpgradeEvent, CancellationEvent],
        name="event_log",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "mrr": lambda ctx: sum(
                _MRR.get(u.plan, 0.0) for u in ctx.agents(SaaSUser)
            ),
            "n_users": lambda ctx: ctx.agent_count(SaaSUser),
            "churned":  lambda ctx: ctx.event_count(CancellationEvent),
        },
        every=30,
        name="monthly_metrics",
    ))

    return sim
