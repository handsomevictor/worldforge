"""supply_chain_world: supply chain inventory simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.distributions import Poisson, Normal, LogNormal
from worldforge.events.base import Event
from worldforge.probes import AggregatorProbe, EventLogProbe


@dataclass
class OrderPlacedEvent(Event):
    retailer_id: str
    supplier_id: str
    qty: int


@dataclass
class StockoutEvent(Event):
    retailer_id: str
    demand: int


@dataclass
class FulfillmentEvent(Event):
    retailer_id: str
    qty: int


class Retailer(Agent):
    inventory: int = field(Normal(mu=100, sigma=20, clip=(10, None)))
    reorder_point: int = field(Normal(mu=30, sigma=10, clip=(5, None)))
    order_qty: int = field(Normal(mu=50, sigma=15, clip=(10, None)))
    demand_lambda: float = field(Poisson(lam=10))
    lead_time_remaining: int = field(0)
    pending_order: int = field(0)

    def step(self, ctx: Any) -> None:
        # Receive pending order
        if self.lead_time_remaining > 0:
            self.lead_time_remaining -= 1
            if self.lead_time_remaining == 0 and self.pending_order > 0:
                self.inventory += self.pending_order
                ctx.emit(FulfillmentEvent(
                    retailer_id=self.id,
                    qty=self.pending_order,
                ))
                self.pending_order = 0

        # Demand realization
        demand = int(Poisson(lam=self.demand_lambda).sample(ctx.rng))
        if demand <= self.inventory:
            self.inventory -= demand
        else:
            ctx.emit(StockoutEvent(retailer_id=self.id, demand=demand))
            self.inventory = 0

        # Reorder trigger
        if self.inventory <= self.reorder_point and self.pending_order == 0:
            self.pending_order = int(self.order_qty)
            self.lead_time_remaining = int(ctx.rng.integers(3, 10))
            ctx.emit(OrderPlacedEvent(
                retailer_id=self.id,
                supplier_id="supplier_1",
                qty=self.pending_order,
            ))


def supply_chain_world(
    n_retailers: int = 50,
    duration_days: int = 365,
    seed: int = 42,
) -> Simulation:
    """Build a supply chain inventory simulation."""
    from worldforge.core.clock import DiscreteClock

    sim = Simulation(
        name="supply_chain_world",
        seed=seed,
        clock=DiscreteClock(steps=duration_days),
    )
    sim.add_agents(Retailer, count=n_retailers)

    sim.add_probe(EventLogProbe(
        events=[OrderPlacedEvent, StockoutEvent, FulfillmentEvent],
        name="event_log",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "avg_inventory":  lambda ctx: ctx.agent_mean(Retailer, "inventory"),
            "n_stockouts":    lambda ctx: ctx.event_count(StockoutEvent),
            "n_orders":       lambda ctx: ctx.event_count(OrderPlacedEvent),
        },
        every=7,
        name="weekly_metrics",
    ))

    return sim
