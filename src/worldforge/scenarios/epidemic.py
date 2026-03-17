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
    state: str = field("S")              # S, I, R
    days_infected: int = field(0)
    recovery_rate: float = field(0.05)   # per-agent field: set via factory in epidemic_world

    def step(self, ctx: Any) -> None:
        if self.state == "I":
            self.days_infected += 1
            # Uses per-instance recovery_rate — no module-level global dependency
            if ctx.rng.random() < self.recovery_rate:
                self.state = "R"
                ctx.emit(RecoveryEvent(
                    person_id=self.id,
                    days_infected=self.days_infected,
                ))


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

    Parameters
    ----------
    population:       Total number of people.
    initial_infected: Number seeded as infected at step 1.
    transmission_prob: Probability of S→I on contact with an infected person.
    recovery_rate:    Probability of I→R per step.
    duration_days:    Number of simulation steps (days).
    seed:             Random seed.

    Returns
    -------
    Simulation object ready to run.
    """
    # All parameters are captured in closures — no module-level globals.
    # This allows multiple epidemic_world() instances to coexist safely
    # in BatchRunner or parallel contexts.
    _transmission_prob = transmission_prob
    _recovery_rate = recovery_rate
    _seeded: list[bool] = [False]

    from worldforge.core.clock import DiscreteClock

    sim = Simulation(
        name="epidemic_world",
        seed=seed,
        clock=DiscreteClock(steps=duration_days),
    )

    # Factory sets per-agent recovery_rate from the closure-local value
    sim.add_agents(
        Person,
        count=population,
        factory=lambda i, rng: Person(recovery_rate=_recovery_rate, _rng=rng),
    )

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

    @sim.global_rule(every=1)
    def spread(ctx: Any) -> None:
        agents = ctx.agents(Person)
        if not agents:
            return

        # Seed initial infections once at step 1
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

        # Collect new infections into a staging dict before applying,
        # preventing double-infection of the same person within one tick.
        infected = [a for a in agents if a.state == "I"]
        rng = ctx.rng
        newly_infected: dict[str, tuple] = {}   # id → (agent, source_id)
        for inf_agent in infected:
            n_contacts = max(1, int(rng.poisson(5)))
            idxs = rng.integers(0, len(agents), size=n_contacts)
            for idx in idxs:
                target = agents[int(idx)]
                if (
                    target.state == "S"
                    and target.id not in newly_infected
                    and rng.random() < _transmission_prob
                ):
                    newly_infected[target.id] = (target, inf_agent.id)

        # Apply atomically — one InfectionEvent per person per tick
        for target, source_id in newly_infected.values():
            target.state = "I"
            ctx.emit(InfectionEvent(person_id=target.id, source_id=source_id))

    return sim
