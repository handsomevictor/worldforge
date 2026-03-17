"""org_dynamics_world: employee organization simulation (hiring, attrition, promotion)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.core.clock import DiscreteClock
from worldforge.distributions import Normal, Categorical, Poisson, Uniform
from worldforge.events.base import Event
from worldforge.probes import EventLogProbe, AggregatorProbe, SnapshotProbe


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class HireEvent(Event):
    employee_id: str
    role: str
    department: str
    salary: float


@dataclass
class PromotionEvent(Event):
    employee_id: str
    old_level: int
    new_level: int
    salary_increase: float


@dataclass
class AttritionEvent(Event):
    employee_id: str
    role: str
    department: str
    tenure_months: int
    reason: str    # "voluntary" | "performance"


@dataclass
class SalaryAdjustmentEvent(Event):
    employee_id: str
    old_salary: float
    new_salary: float


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Employee(Agent):
    role: str = field(Categorical(
        choices=["engineer", "designer", "manager", "sales", "ops"],
        weights=[0.35, 0.15, 0.20, 0.20, 0.10],
    ))
    department: str = field(Categorical(
        choices=["engineering", "product", "sales", "operations", "finance"],
        weights=[0.40, 0.20, 0.20, 0.10, 0.10],
    ))
    level: int = field(1)           # 1=junior, 2=mid, 3=senior, 4=lead, 5=principal
    tenure_months: int = field(0)
    salary: float = field(Normal(mu=80_000, sigma=20_000, clip=(30_000, None)))
    performance: float = field(Normal(mu=0.70, sigma=0.15, clip=(0, 1)))
    engagement: float = field(Normal(mu=0.75, sigma=0.15, clip=(0, 1)))
    attrition_risk: float = field(0.0)

    def step(self, ctx: Any) -> None:
        self.tenure_months += 1
        self._update_engagement(ctx)
        self._update_performance(ctx)
        self._update_attrition_risk()

        # Voluntary / involuntary attrition
        if ctx.rng.random() < self.attrition_risk:
            reason = "performance" if self.performance < 0.30 else "voluntary"
            ctx.emit(AttritionEvent(
                employee_id=self.id,
                role=self.role,
                department=self.department,
                tenure_months=self.tenure_months,
                reason=reason,
            ))
            ctx.remove_agent(self)
            return

        # Annual merit raise (every 12 months)
        if self.tenure_months % 12 == 0 and self.tenure_months > 0:
            if self.performance >= 0.60:
                pct = float(Uniform(0.02, 0.10).sample(ctx.rng)) * self.performance
                new_salary = self.salary * (1 + pct)
                ctx.emit(SalaryAdjustmentEvent(
                    employee_id=self.id,
                    old_salary=self.salary,
                    new_salary=new_salary,
                ))
                self.salary = new_salary

        # Promotion check (every 18 months, high performers)
        if (
            self.tenure_months % 18 == 0
            and self.level < 5
            and self.performance >= 0.75
            and ctx.rng.random() < 0.25
        ):
            bump = self.salary * float(Uniform(0.10, 0.20).sample(ctx.rng))
            ctx.emit(PromotionEvent(
                employee_id=self.id,
                old_level=self.level,
                new_level=self.level + 1,
                salary_increase=bump,
            ))
            self.level += 1
            self.salary += bump
            self.engagement = min(1.0, self.engagement + 0.15)

    def _update_engagement(self, ctx: Any) -> None:
        # Engagement drifts naturally; slight decay over time
        drift = float(Normal(mu=-0.002, sigma=0.01).sample(ctx.rng))
        self.engagement = max(0.0, min(1.0, self.engagement + drift))

    def _update_performance(self, ctx: Any) -> None:
        # Performance correlates with engagement
        drift = float(Normal(mu=0.0, sigma=0.02).sample(ctx.rng))
        self.performance = max(0.0, min(1.0, self.performance + drift))
        # Low engagement depresses performance
        if self.engagement < 0.40:
            self.performance = max(0.0, self.performance - 0.01)

    def _update_attrition_risk(self) -> None:
        base = 0.005
        # Low engagement → higher risk
        if self.engagement < 0.40:
            base += 0.020
        # Low performance → performance management risk
        if self.performance < 0.30:
            base += 0.030
        # Early tenure bump (first 6 months: new-hire jitters)
        if self.tenure_months < 6:
            base += 0.010
        # Long tenure bump (burnout after 5+ years)
        if self.tenure_months > 60:
            base += 0.008
        self.attrition_risk = min(base, 0.08)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def org_dynamics_world(
    n_employees: int = 500,
    steps: int = 60,           # months
    hiring_rate: float = 0.03, # fraction of headcount hired per step
    seed: int = 42,
) -> Simulation:
    """
    Build an employee organization dynamics simulation.

    Each step represents one month.

    Parameters
    ----------
    n_employees:  Starting headcount.
    steps:        Number of months to simulate.
    hiring_rate:  Monthly hiring rate as fraction of current headcount.
    seed:         Random seed.

    Returns
    -------
    Simulation object ready to run.

    Example
    -------
    >>> sim = org_dynamics_world(n_employees=200, steps=24)
    >>> result = sim.run()
    >>> df = result.to_pandas()["org_metrics"]
    """
    sim = Simulation(
        name="org_dynamics_world",
        seed=seed,
        clock=DiscreteClock(steps=steps),
    )
    sim.add_agents(Employee, count=n_employees)

    @sim.global_rule(every=1)
    def backfill_hiring(ctx: Any) -> None:
        """Hire to replace attrition and maintain growth target."""
        headcount = ctx.agent_count(Employee)
        n_hire = max(0, int(headcount * hiring_rate))

        def _emit_hire(emp: Any) -> None:
            ctx.emit(HireEvent(
                employee_id=emp.id,
                role=emp.role,
                department=emp.department,
                salary=emp.salary,
            ))

        for _ in range(n_hire):
            # Use init callback so HireEvent is emitted immediately after spawn
            ctx.spawn(Employee, count=1, init=_emit_hire)

    sim.add_probe(EventLogProbe(
        events=[HireEvent, PromotionEvent, AttritionEvent, SalaryAdjustmentEvent],
        name="event_log",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "headcount": lambda ctx: ctx.agent_count(Employee),
            "avg_salary": lambda ctx: ctx.agent_mean(Employee, "salary"),
            "avg_performance": lambda ctx: ctx.agent_mean(Employee, "performance"),
            "avg_engagement": lambda ctx: ctx.agent_mean(Employee, "engagement"),
            "avg_level": lambda ctx: ctx.agent_mean(Employee, "level"),
            "attritions": lambda ctx: ctx.event_count(AttritionEvent),
            "promotions": lambda ctx: ctx.event_count(PromotionEvent),
            "eng_headcount": lambda ctx: sum(
                1 for e in ctx.agents(Employee) if e.department == "engineering"
            ),
        },
        every=1,
        name="org_metrics",
    ))
    sim.add_probe(SnapshotProbe(
        agent_type=Employee,
        fields=["id", "role", "department", "level", "salary", "tenure_months",
                "performance", "engagement"],
        every=12,   # annual snapshot
        sample_rate=0.3,
        name="employee_snapshot",
    ))

    return sim
