"""market_microstructure_world: limit order book simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.distributions import Normal, LogNormal, Uniform
from worldforge.environments.market import MarketEnvironment, Trade
from worldforge.events.base import Event
from worldforge.probes import AggregatorProbe, TimeSeriesProbe


@dataclass
class OrderEvent(Event):
    trader_id: str
    asset: str
    side: str
    price: float
    qty: float


class MarketMaker(Agent):
    """Provides liquidity by posting both bid and ask orders."""
    spread: float = field(Normal(mu=1.0, sigma=0.2, clip=(0.1, None)))
    qty_per_order: float = field(Uniform(1, 10))

    def step(self, ctx: Any) -> None:
        env: MarketEnvironment = ctx.environment
        if not isinstance(env, MarketEnvironment):
            return
        asset = "STOCK"
        mid = env.mid_price(asset)
        half = self.spread / 2
        # Post bid and ask
        for side, price in [("buy", mid - half), ("sell", mid + half)]:
            trades = env.submit_order(self.id, asset, side, price, self.qty_per_order)
            ctx.emit(OrderEvent(
                trader_id=self.id, asset=asset, side=side,
                price=price, qty=self.qty_per_order,
            ))


class NoiseTrader(Agent):
    """Submits random orders near the current price."""

    def step(self, ctx: Any) -> None:
        env: MarketEnvironment = ctx.environment
        if not isinstance(env, MarketEnvironment):
            return
        if ctx.rng.random() > 0.3:
            return
        asset = "STOCK"
        mid = env.mid_price(asset)
        side = "buy" if ctx.rng.random() > 0.5 else "sell"
        noise = float(Normal(mu=0, sigma=2).sample(ctx.rng))
        price = max(0.01, mid + noise)
        qty = float(Uniform(1, 5).sample(ctx.rng))
        env.submit_order(self.id, asset, side, price, qty)
        ctx.emit(OrderEvent(
            trader_id=self.id, asset=asset, side=side, price=price, qty=qty,
        ))


def market_microstructure_world(
    n_market_makers: int = 5,
    n_noise_traders: int = 50,
    initial_price: float = 100.0,
    duration_steps: int = 500,
    seed: int = 42,
) -> Simulation:
    """Build a market microstructure (order book) simulation."""
    from worldforge.core.clock import DiscreteClock

    env = MarketEnvironment(
        assets=["STOCK"],
        initial_prices={"STOCK": initial_price},
        tick_size=0.01,
    )

    sim = Simulation(
        name="market_microstructure",
        seed=seed,
        clock=DiscreteClock(steps=duration_steps),
    )
    sim.add_agents(MarketMaker, count=n_market_makers)
    sim.add_agents(NoiseTrader, count=n_noise_traders)

    sim.add_probe(TimeSeriesProbe(
        series={"mid_price": lambda ctx: (
            ctx.environment.mid_price("STOCK")
            if isinstance(ctx.environment, MarketEnvironment) else 0.0
        )},
        every=1,
        name="price_series",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "n_trades": lambda ctx: len(
                ctx.environment.trade_history("STOCK")
                if isinstance(ctx.environment, MarketEnvironment) else []
            ),
        },
        every=10,
        name="market_metrics",
    ))

    # NOTE: the runner must set sim._ctx.environment = env for env queries to work.
    # We store env on the sim for the user to wire up manually.
    sim._market_env = env

    return sim
