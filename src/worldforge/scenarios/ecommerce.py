"""ecommerce_world: e-commerce user behavior simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.core.clock import DiscreteClock
from worldforge.distributions import (
    Normal, LogNormal, Categorical, Exponential, Uniform,
)
from worldforge.events.base import Event
from worldforge.probes import EventLogProbe, SnapshotProbe, AggregatorProbe
from worldforge.time.calendar import CalendarClock


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class PurchaseEvent(Event):
    user_id: str
    product_id: str
    amount: float
    category: str


@dataclass
class ChurnEvent(Event):
    user_id: str
    tenure_days: int


@dataclass
class UserSignupEvent(Event):
    user_id: str
    tier: str


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class EcommerceUser(Agent):
    balance: float = field(Normal(mu=3000, sigma=1500, clip=(0, None)))
    tier: str = field(Categorical(
        choices=["free", "premium", "vip"],
        weights=[0.70, 0.25, 0.05],
    ))
    age_days: int = field(0)
    churn_risk: float = field(0.0)
    is_active: bool = field(True)

    # Purchase rate by tier (per step)
    _purchase_rate = {"free": 0.02, "premium": 0.06, "vip": 0.12}
    _avg_order_value = {"free": 50, "premium": 150, "vip": 500}

    def step(self, ctx: Any) -> None:
        if not self.is_active:
            return
        self.age_days += 1
        self._update_churn_risk()

        rate = self._purchase_rate[self.tier]
        if ctx.rng.random() < rate and self.balance >= 10:
            amount = float(
                Normal(
                    mu=self._avg_order_value[self.tier],
                    sigma=self._avg_order_value[self.tier] * 0.3,
                    clip=(1, self.balance),
                ).sample(ctx.rng)
            )
            cat = ctx.rng.choice(["electronics", "clothing", "food", "home"])
            self.balance -= amount
            ctx.emit(PurchaseEvent(
                user_id=self.id,
                product_id=f"prod_{ctx.rng.integers(1, 1000)}",
                amount=amount,
                category=cat,
            ))

        if ctx.rng.random() < self.churn_risk:
            ctx.emit(ChurnEvent(user_id=self.id, tenure_days=self.age_days))
            ctx.remove_agent(self)

    def _update_churn_risk(self) -> None:
        base = 0.001
        if self.tier == "free" and self.age_days > 30:
            base += 0.003
        if self.balance < 100:
            base += 0.005
        self.churn_risk = min(base, 0.1)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def ecommerce_world(
    n_users: int = 1000,
    duration: str = "90 days",
    seed: int = 42,
    seasonality: bool = False,
    include_fraud: bool = False,
) -> Simulation:
    """
    Build a ready-to-run e-commerce simulation.

    Parameters
    ----------
    n_users:      Number of users to simulate.
    duration:     Calendar duration string like '90 days' or '1 year'.
    seed:         Random seed.
    seasonality:  Reserved for future use.
    include_fraud: Reserved for future use.

    Returns
    -------
    Simulation object (not yet run; call .run() on it).
    """
    from worldforge.time.calendar import CalendarClock, parse_duration
    from datetime import datetime, timedelta

    start_dt = datetime(2024, 1, 1)
    duration_td = parse_duration(duration)
    end_dt = start_dt + duration_td

    clock = CalendarClock(
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        step="1 day",
    )

    sim = Simulation(name="ecommerce_world", seed=seed, clock=clock)
    sim.add_agents(EcommerceUser, count=n_users)

    # Probes
    sim.add_probe(EventLogProbe(
        events=[PurchaseEvent, ChurnEvent, UserSignupEvent],
        name="event_log",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "dau": lambda ctx: ctx.agent_count(EcommerceUser),
            "gmv_daily": lambda ctx: ctx.event_sum(PurchaseEvent, "amount"),
            "churned": lambda ctx: ctx.event_count(ChurnEvent),
        },
        every="1 day",
        name="daily_metrics",
    ))
    sim.add_probe(SnapshotProbe(
        agent_type=EcommerceUser,
        fields=["id", "balance", "tier", "age_days", "churn_risk"],
        every="1 week",
        sample_rate=0.1,
        name="user_snapshot",
    ))

    return sim
