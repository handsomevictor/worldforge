"""rideshare_world: ride-sharing platform simulation (drivers + riders, dynamic pricing)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.core.clock import DiscreteClock
from worldforge.distributions import Normal, Categorical, Poisson, Uniform
from worldforge.events.base import Event
from worldforge.probes import EventLogProbe, AggregatorProbe


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class RideRequestEvent(Event):
    rider_id: str
    distance_km: float
    surge_multiplier: float


@dataclass
class RideCompletedEvent(Event):
    rider_id: str
    driver_id: str
    fare: float
    distance_km: float
    wait_steps: int


@dataclass
class RideCancelledEvent(Event):
    rider_id: str
    reason: str  # "no_driver" | "timeout"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Driver(Agent):
    status: str = field("idle")            # idle | busy
    trips_completed: int = field(0)
    earnings: float = field(0.0)
    rating: float = field(Normal(mu=4.5, sigma=0.3, clip=(1.0, 5.0)))
    online: bool = field(True)             # whether the driver is logged on

    def step(self, ctx: Any) -> None:
        # Drivers randomly go offline for rest (5% chance per step if idle)
        if self.status == "idle" and ctx.rng.random() < 0.05:
            self.online = False
        if not self.online:
            # Come back online with 20% probability
            if ctx.rng.random() < 0.20:
                self.online = True
            return

        # Busy drivers become idle after one step (each trip = 1 step)
        if self.status == "busy":
            self.status = "idle"


class Rider(Agent):
    patience_steps: int = field(Poisson(lam=3))   # max steps willing to wait
    wait_steps: int = field(0)
    has_active_request: bool = field(False)
    total_rides: int = field(0)
    total_spent: float = field(0.0)
    _pending_request: dict | None = None           # non-field internal state

    def on_born(self, ctx: Any) -> None:
        self._pending_request = None

    def step(self, ctx: Any) -> None:
        if not self.has_active_request:
            # Each step, a rider has a 15% chance of wanting a ride
            if ctx.rng.random() < 0.15:
                distance_km = float(Normal(mu=8, sigma=4, clip=(0.5, 50)).sample(ctx.rng))
                surge = _compute_surge(ctx)
                ctx.emit(RideRequestEvent(
                    rider_id=self.id,
                    distance_km=distance_km,
                    surge_multiplier=surge,
                ))
                self.has_active_request = True
                self.wait_steps = 0
                self._pending_request = {"distance_km": distance_km, "surge": surge}
        else:
            self.wait_steps += 1
            # Cancel if waiting too long
            if self.wait_steps > self.patience_steps:
                ctx.emit(RideCancelledEvent(rider_id=self.id, reason="timeout"))
                self.has_active_request = False
                self._pending_request = None


# ---------------------------------------------------------------------------
# Module-level state for global rule
# ---------------------------------------------------------------------------

_base_fare_per_km = 1.5
_surge_threshold = 0.5   # ratio of riders/drivers above which surge kicks in


def _compute_surge(ctx: Any) -> float:
    """Dynamic pricing: surge multiplier based on demand/supply ratio."""
    n_drivers = sum(1 for d in ctx.agents(Driver) if d.status == "idle" and d.online)
    n_riders_waiting = sum(1 for r in ctx.agents(Rider) if r.has_active_request)
    if n_drivers == 0:
        return 3.0
    ratio = n_riders_waiting / max(n_drivers, 1)
    if ratio > 2.0:
        return 2.5
    if ratio > 1.0:
        return 1.5
    return 1.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def rideshare_world(
    n_drivers: int = 200,
    n_riders: int = 1000,
    steps: int = 200,
    seed: int = 42,
) -> Simulation:
    """
    Build a ride-sharing platform simulation.

    Each step represents ~5 minutes of real time.

    Parameters
    ----------
    n_drivers:  Number of driver agents.
    n_riders:   Number of rider agents.
    steps:      Number of simulation steps (each ~5 min).
    seed:       Random seed.

    Returns
    -------
    Simulation object ready to run.

    Example
    -------
    >>> sim = rideshare_world(n_drivers=100, n_riders=500, steps=100)
    >>> result = sim.run()
    >>> df = result.to_pandas()["daily_metrics"]
    """
    sim = Simulation(
        name="rideshare_world",
        seed=seed,
        clock=DiscreteClock(steps=steps),
    )
    sim.add_agents(Driver, count=n_drivers)
    sim.add_agents(Rider, count=n_riders)

    @sim.global_rule(every=1)
    def match_rides(ctx: Any) -> None:
        """Match waiting riders to idle drivers, complete rides."""
        idle_drivers = [d for d in ctx.agents(Driver) if d.status == "idle" and d.online]
        waiting_riders = [r for r in ctx.agents(Rider) if r.has_active_request]

        for rider, driver in zip(waiting_riders, idle_drivers):
            req = rider._pending_request
            if req is None:
                continue
            distance_km = req["distance_km"]
            surge = req["surge"]
            fare = round(distance_km * _base_fare_per_km * surge, 2)

            ctx.emit(RideCompletedEvent(
                rider_id=rider.id,
                driver_id=driver.id,
                fare=fare,
                distance_km=distance_km,
                wait_steps=rider.wait_steps,
            ))

            # Update state
            driver.status = "busy"
            driver.trips_completed += 1
            driver.earnings += fare * 0.80   # driver keeps 80%

            rider.has_active_request = False
            rider._pending_request = None
            rider.total_rides += 1
            rider.total_spent += fare

    sim.add_probe(EventLogProbe(
        events=[RideCompletedEvent, RideCancelledEvent, RideRequestEvent],
        name="event_log",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "rides_completed": lambda ctx: ctx.event_count(RideCompletedEvent),
            "rides_cancelled": lambda ctx: ctx.event_count(RideCancelledEvent),
            "gmv": lambda ctx: ctx.event_sum(RideCompletedEvent, "fare"),
            "idle_drivers": lambda ctx: sum(
                1 for d in ctx.agents(Driver) if d.status == "idle" and d.online
            ),
            "waiting_riders": lambda ctx: sum(
                1 for r in ctx.agents(Rider) if r.has_active_request
            ),
            "surge_multiplier": lambda ctx: _compute_surge(ctx),
        },
        every=1,
        name="platform_metrics",
    ))

    return sim
