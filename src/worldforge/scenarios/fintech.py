"""fintech_world: fintech/banking user simulation with loan and savings behavior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.distributions import Normal, Categorical, LogNormal
from worldforge.events.base import Event
from worldforge.probes import AggregatorProbe, EventLogProbe, SnapshotProbe


@dataclass
class LoanApplicationEvent(Event):
    user_id: str
    amount: float
    approved: bool


@dataclass
class DefaultEvent(Event):
    user_id: str
    outstanding: float


@dataclass
class DepositEvent(Event):
    user_id: str
    amount: float


class BankUser(Agent):
    balance: float = field(Normal(mu=5000, sigma=3000, clip=(0, None)))
    credit_score: int = field(Normal(mu=650, sigma=100, clip=(300, 850)))
    income: float = field(LogNormal(mu=10.5, sigma=0.5))  # monthly income
    loan_balance: float = field(0.0)
    tier: str = field(Categorical(
        choices=["retail", "premium", "private"],
        weights=[0.75, 0.20, 0.05],
    ))
    days: int = field(0)

    def step(self, ctx: Any) -> None:
        self.days += 1

        # Monthly salary deposit
        if self.days % 30 == 0:
            deposit = float(Normal(mu=self.income, sigma=self.income * 0.05).sample(ctx.rng))
            self.balance += deposit
            ctx.emit(DepositEvent(user_id=self.id, amount=deposit))

        # Occasional loan application
        if self.loan_balance == 0 and ctx.rng.random() < 0.002:
            loan_amount = float(Normal(mu=10000, sigma=5000, clip=(1000, None)).sample(ctx.rng))
            approved = self.credit_score >= 600 and self.income > 2000
            ctx.emit(LoanApplicationEvent(
                user_id=self.id,
                amount=loan_amount,
                approved=approved,
            ))
            if approved:
                self.loan_balance = loan_amount
                self.balance += loan_amount

        # Loan repayment
        if self.loan_balance > 0 and self.days % 30 == 0:
            payment = min(self.loan_balance, self.loan_balance * 0.05)
            if self.balance >= payment:
                self.balance -= payment
                self.loan_balance -= payment
            else:
                ctx.emit(DefaultEvent(user_id=self.id, outstanding=self.loan_balance))
                self.loan_balance = 0


def fintech_world(
    n_users: int = 5000,
    duration_days: int = 365,
    seed: int = 42,
) -> Simulation:
    """Build a fintech/banking simulation."""
    from worldforge.core.clock import DiscreteClock

    sim = Simulation(
        name="fintech_world",
        seed=seed,
        clock=DiscreteClock(steps=duration_days),
    )
    sim.add_agents(BankUser, count=n_users)

    sim.add_probe(EventLogProbe(
        events=[LoanApplicationEvent, DefaultEvent, DepositEvent],
        name="event_log",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            # last=30 scopes to the current 30-step window — truly "this month"
            "deposits_this_month": lambda ctx: ctx.event_sum(DepositEvent, "amount", last=30),
            "loans_this_month":    lambda ctx: ctx.event_count(LoanApplicationEvent, last=30),
            "defaults_this_month": lambda ctx: ctx.event_count(DefaultEvent, last=30),
            # Cumulative totals since simulation start
            "total_deposits":      lambda ctx: ctx.event_sum(DepositEvent, "amount"),
            "total_loans":         lambda ctx: ctx.event_count(LoanApplicationEvent),
            "total_defaults":      lambda ctx: ctx.event_count(DefaultEvent),
            "avg_balance":         lambda ctx: ctx.agent_mean(BankUser, "balance"),
        },
        every=30,
        name="monthly_metrics",
    ))
    sim.add_probe(SnapshotProbe(
        agent_type=BankUser,
        fields=["id", "balance", "credit_score", "tier", "loan_balance"],
        every=30,
        sample_rate=0.05,
        name="user_snapshot",
    ))

    return sim
