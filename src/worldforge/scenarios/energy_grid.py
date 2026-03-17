"""energy_grid_world: power grid simulation (generators, consumers, storage)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.core.clock import DiscreteClock
from worldforge.distributions import Normal, Categorical, Poisson, Uniform
from worldforge.events.base import Event
from worldforge.probes import EventLogProbe, AggregatorProbe, TimeSeriesProbe


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class PowerShortageEvent(Event):
    deficit_mw: float
    load_shed_mw: float


@dataclass
class GeneratorTripEvent(Event):
    generator_id: str
    gen_type: str
    capacity_lost_mw: float
    reason: str  # "fault" | "maintenance"


@dataclass
class StorageChargeEvent(Event):
    storage_id: str
    action: str   # "charge" | "discharge"
    energy_mwh: float


@dataclass
class RenewableSpillEvent(Event):
    generator_id: str
    spilled_mw: float


# ---------------------------------------------------------------------------
# Module-level demand profile (24-hour pattern, index = hour)
# ---------------------------------------------------------------------------

_DEMAND_PATTERN = [
    0.55, 0.50, 0.48, 0.47, 0.48, 0.52,   # 00:00–05:00 (night trough)
    0.62, 0.75, 0.88, 0.92, 0.94, 0.95,   # 06:00–11:00 (morning ramp)
    0.97, 0.98, 0.96, 0.94, 0.95, 1.00,   # 12:00–17:00 (day peak)
    0.98, 0.96, 0.90, 0.82, 0.72, 0.62,   # 18:00–23:00 (evening decline)
]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Generator(Agent):
    gen_type: str = field(Categorical(
        choices=["coal", "gas", "nuclear", "wind", "solar"],
        weights=[0.20, 0.30, 0.15, 0.20, 0.15],
    ))
    capacity_mw: float = field(Normal(mu=500, sigma=200, clip=(50, 2000)))
    output_mw: float = field(0.0)
    online: bool = field(True)
    maintenance_countdown: int = field(0)

    # Fuel cost per MWh by type
    _fuel_cost = {"coal": 40, "gas": 60, "nuclear": 20, "wind": 5, "solar": 3}
    # Forced outage probability per step
    _outage_prob = {"coal": 0.002, "gas": 0.001, "nuclear": 0.0005, "wind": 0.003, "solar": 0.001}

    def step(self, ctx: Any) -> None:
        if self.maintenance_countdown > 0:
            self.maintenance_countdown -= 1
            self.online = False
            self.output_mw = 0.0
            if self.maintenance_countdown == 0:
                self.online = True
            return

        # Random forced outage
        if self.online and ctx.rng.random() < self._outage_prob[self.gen_type]:
            self.online = False
            self.maintenance_countdown = int(ctx.rng.integers(3, 12))
            ctx.emit(GeneratorTripEvent(
                generator_id=self.id,
                gen_type=self.gen_type,
                capacity_lost_mw=self.output_mw,
                reason="fault",
            ))
            self.output_mw = 0.0
            return

        if not self.online:
            self.output_mw = 0.0
            return

        # Output calculation
        step_hour = int(ctx.now) % 24
        if self.gen_type == "solar":
            # Solar only produces during daylight hours (6–18)
            if 6 <= step_hour <= 18:
                solar_factor = 1.0 - abs(step_hour - 12) / 8.0
                self.output_mw = self.capacity_mw * solar_factor * float(
                    Normal(mu=1.0, sigma=0.1, clip=(0, 1)).sample(ctx.rng)
                )
            else:
                self.output_mw = 0.0
        elif self.gen_type == "wind":
            # Wind is variable
            self.output_mw = self.capacity_mw * float(
                Normal(mu=0.35, sigma=0.20, clip=(0, 1)).sample(ctx.rng)
            )
        else:
            # Dispatchable: run at full capacity
            self.output_mw = self.capacity_mw


class Consumer(Agent):
    base_load_mw: float = field(Normal(mu=50, sigma=20, clip=(5, 200)))
    consumer_type: str = field(Categorical(
        choices=["residential", "commercial", "industrial"],
        weights=[0.50, 0.30, 0.20],
    ))
    current_load_mw: float = field(0.0)

    def step(self, ctx: Any) -> None:
        step_hour = int(ctx.now) % 24
        demand_factor = _DEMAND_PATTERN[step_hour]

        # Industrial consumers have flatter demand profile
        if self.consumer_type == "industrial":
            demand_factor = 0.85 + ctx.rng.random() * 0.15

        noise = float(Normal(mu=1.0, sigma=0.05, clip=(0.5, 1.5)).sample(ctx.rng))
        self.current_load_mw = self.base_load_mw * demand_factor * noise


class BatteryStorage(Agent):
    capacity_mwh: float = field(Normal(mu=200, sigma=50, clip=(50, 500)))
    charge_rate_mw: float = field(Normal(mu=50, sigma=10, clip=(10, 100)))
    charge_level_mwh: float = field(0.0)
    efficiency: float = field(Normal(mu=0.92, sigma=0.02, clip=(0.85, 0.98)))

    def on_born(self, ctx: Any) -> None:
        # Start half-charged
        self.charge_level_mwh = self.capacity_mwh * 0.5

    def step(self, ctx: Any) -> None:
        pass   # Storage is dispatched by global_rule


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def energy_grid_world(
    n_generators: int = 20,
    n_consumers: int = 100,
    n_storage: int = 5,
    steps: int = 168,   # one week at hourly resolution
    seed: int = 42,
) -> Simulation:
    """
    Build an electrical power grid simulation.

    Each step represents one hour.

    Parameters
    ----------
    n_generators: Number of generator agents (mix of coal/gas/nuclear/wind/solar).
    n_consumers:  Number of consumer agents.
    n_storage:    Number of battery storage units.
    steps:        Hours to simulate (default 168 = 1 week).
    seed:         Random seed.

    Returns
    -------
    Simulation object ready to run.

    Example
    -------
    >>> sim = energy_grid_world(n_generators=10, n_consumers=50, steps=24)
    >>> result = sim.run()
    >>> df = result.to_pandas()["grid_timeseries"]
    """
    sim = Simulation(
        name="energy_grid_world",
        seed=seed,
        clock=DiscreteClock(steps=steps),
    )
    sim.add_agents(Generator, count=n_generators)
    sim.add_agents(Consumer, count=n_consumers)
    sim.add_agents(BatteryStorage, count=n_storage)

    @sim.global_rule(every=1)
    def dispatch_and_balance(ctx: Any) -> None:
        """
        Balance supply and demand each hour.
        Use storage to fill gaps; shed load or spill renewables as needed.
        """
        total_supply = sum(g.output_mw for g in ctx.agents(Generator))
        total_demand = sum(c.current_load_mw for c in ctx.agents(Consumer))
        storage_units = ctx.agents(BatteryStorage)

        net = total_supply - total_demand

        if net > 0:
            # Excess power: charge batteries or spill renewables
            excess = net
            for bat in storage_units:
                if bat.charge_level_mwh < bat.capacity_mwh and excess > 0:
                    charge = min(bat.charge_rate_mw, excess,
                                 bat.capacity_mwh - bat.charge_level_mwh)
                    bat.charge_level_mwh += charge * bat.efficiency
                    excess -= charge
                    ctx.emit(StorageChargeEvent(
                        storage_id=bat.id,
                        action="charge",
                        energy_mwh=charge,
                    ))

            # Any remaining excess is spilled from renewables
            if excess > 0:
                for gen in ctx.agents(Generator):
                    if gen.gen_type in ("wind", "solar") and gen.output_mw > 0 and excess > 0:
                        spill = min(gen.output_mw * 0.5, excess)
                        ctx.emit(RenewableSpillEvent(
                            generator_id=gen.id,
                            spilled_mw=spill,
                        ))
                        excess -= spill
                        if excess <= 0:
                            break

        else:
            # Deficit: discharge batteries
            deficit = abs(net)
            for bat in storage_units:
                if bat.charge_level_mwh > 0 and deficit > 0:
                    discharge = min(bat.charge_rate_mw, deficit, bat.charge_level_mwh)
                    bat.charge_level_mwh -= discharge
                    deficit -= discharge
                    ctx.emit(StorageChargeEvent(
                        storage_id=bat.id,
                        action="discharge",
                        energy_mwh=discharge,
                    ))

            # If still a deficit, load shedding
            if deficit > 0.5:
                load_shed = deficit * 0.8
                ctx.emit(PowerShortageEvent(
                    deficit_mw=deficit,
                    load_shed_mw=load_shed,
                ))

    sim.add_probe(EventLogProbe(
        events=[PowerShortageEvent, GeneratorTripEvent, StorageChargeEvent],
        name="event_log",
    ))
    sim.add_probe(TimeSeriesProbe(
        series={
            "total_supply_mw": lambda ctx: sum(g.output_mw for g in ctx.agents(Generator)),
            "total_demand_mw": lambda ctx: sum(c.current_load_mw for c in ctx.agents(Consumer)),
            "storage_level_mwh": lambda ctx: sum(
                b.charge_level_mwh for b in ctx.agents(BatteryStorage)
            ),
            "online_generators": lambda ctx: sum(
                1 for g in ctx.agents(Generator) if g.online
            ),
        },
        every=1,
        name="grid_timeseries",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "shortage_events": lambda ctx: ctx.event_count(PowerShortageEvent),
            "generator_trips": lambda ctx: ctx.event_count(GeneratorTripEvent),
            "renewable_spill_mw": lambda ctx: ctx.event_sum(RenewableSpillEvent, "spilled_mw"),
            "total_supply_mw": lambda ctx: sum(g.output_mw for g in ctx.agents(Generator)),
            "total_demand_mw": lambda ctx: sum(c.current_load_mw for c in ctx.agents(Consumer)),
        },
        every=24,   # daily summary
        name="daily_grid_summary",
    ))

    return sim
