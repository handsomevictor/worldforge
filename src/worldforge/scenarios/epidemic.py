"""epidemic_world: SIR epidemic simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.events.base import Event
from worldforge.probes import AggregatorProbe, EventLogProbe


@dataclass
class InfectionEvent(Event):
    person_id: str
    source_id: str | None


@dataclass
class RecoveryEvent(Event):
    person_id: str
    days_infected: int


class Person(Agent):
    state: str = field("S")          # S, I, R
    days_infected: int = field(0)

    def step(self, ctx: Any) -> None:
        if self.state == "I":
            self.days_infected += 1
            if ctx.rng.random() < _recovery_rate:
                self.state = "R"
                ctx.emit(RecoveryEvent(
                    person_id=self.id,
                    days_infected=self.days_infected,
                ))


# module-level mutable so global_rule closure can read it
_recovery_rate = 0.05
_transmission_prob = 0.3


def epidemic_world(
    population: int = 10_000,
    initial_infected: int = 10,
    transmission_prob: float = 0.3,
    recovery_rate: float = 0.05,
    duration_days: int = 180,
    seed: int = 42,
) -> Simulation:
    """
    Build an SIR epidemic simulation.

    Returns a Simulation object ready to run.
    """
    global _recovery_rate, _transmission_prob
    _recovery_rate = recovery_rate
    _transmission_prob = transmission_prob

    from worldforge.core.clock import DiscreteClock

    sim = Simulation(
        name="epidemic_world",
        seed=seed,
        clock=DiscreteClock(steps=duration_days),
    )
    sim.add_agents(Person, count=population)

    sim.add_probe(AggregatorProbe(
        metrics={
            "S": lambda ctx: sum(1 for p in ctx.agents(Person) if p.state == "S"),
            "I": lambda ctx: sum(1 for p in ctx.agents(Person) if p.state == "I"),
            "R": lambda ctx: sum(1 for p in ctx.agents(Person) if p.state == "R"),
        },
        every=1,
        name="sir_curve",
    ))
    sim.add_probe(EventLogProbe(
        events=[InfectionEvent, RecoveryEvent],
        name="event_log",
    ))

    _seeded: list[bool] = [False]

    @sim.global_rule(every=1)
    def spread(ctx: Any) -> None:
        agents = ctx.agents(Person)
        if not agents:
            return

        # Seed initial infections once
        if not _seeded[0]:
            _seeded[0] = True
            rng = ctx.rng
            k = min(initial_infected, len(agents))
            idxs = rng.choice(len(agents), size=k, replace=False)
            for i in idxs:
                a = agents[int(i)]
                if a.state == "S":
                    a.state = "I"
                    ctx.emit(InfectionEvent(person_id=a.id, source_id=None))

        # Spreading step
        infected = [a for a in agents if a.state == "I"]
        rng = ctx.rng
        for inf_agent in infected:
            n_contacts = max(1, int(rng.poisson(5)))
            idxs = rng.integers(0, len(agents), size=n_contacts)
            for idx in idxs:
                target = agents[int(idx)]
                if target.state == "S" and rng.random() < _transmission_prob:
                    target.state = "I"
                    ctx.emit(InfectionEvent(
                        person_id=target.id,
                        source_id=inf_agent.id,
                    ))

    return sim
