"""game_economy_world: virtual game economy simulation (players + item market)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.core.clock import DiscreteClock
from worldforge.distributions import Normal, Categorical, Poisson, LogNormal, Uniform
from worldforge.events.base import Event
from worldforge.probes import EventLogProbe, AggregatorProbe, SnapshotProbe


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class ItemPurchaseEvent(Event):
    player_id: str
    item_type: str
    price: float
    is_real_money: bool    # True = IAP, False = in-game gold


@dataclass
class PlayerLevelUpEvent(Event):
    player_id: str
    new_level: int


@dataclass
class PlayerChurnEvent(Event):
    player_id: str
    total_sessions: int
    lifetime_spend: float


@dataclass
class MarketListingEvent(Event):
    seller_id: str
    item_type: str
    list_price: float


# ---------------------------------------------------------------------------
# Global market prices (mutable, updated by global_rule)
# ---------------------------------------------------------------------------

_market_prices: dict[str, float] = {
    "sword":   100.0,
    "shield":  80.0,
    "potion":  20.0,
    "armor":   150.0,
    "gem":     500.0,
}

_ITEM_TYPES = list(_market_prices.keys())


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Player(Agent):
    gold: float = field(Normal(mu=500, sigma=200, clip=(0, None)))
    level: int = field(1)
    sessions: int = field(0)
    lifetime_spend: float = field(0.0)
    segment: str = field(Categorical(
        choices=["free", "minnow", "dolphin", "whale"],
        weights=[0.60, 0.25, 0.12, 0.03],
    ))
    churn_risk: float = field(0.0)

    # IAP conversion rates by segment
    _iap_rate = {"free": 0.0, "minnow": 0.02, "dolphin": 0.08, "whale": 0.20}
    # Gold earn rate per session by level (higher level = richer)
    _gold_earn_rate = 10.0

    def step(self, ctx: Any) -> None:
        self.sessions += 1
        self._gain_gold_and_xp(ctx)
        self._maybe_purchase_item(ctx)
        self._maybe_level_up(ctx)
        self._update_churn_risk(ctx)
        if ctx.rng.random() < self.churn_risk:
            ctx.emit(PlayerChurnEvent(
                player_id=self.id,
                total_sessions=self.sessions,
                lifetime_spend=self.lifetime_spend,
            ))
            ctx.remove_agent(self)

    def _gain_gold_and_xp(self, ctx: Any) -> None:
        earn = self._gold_earn_rate * self.level * float(
            Normal(mu=1.0, sigma=0.2, clip=(0.1, None)).sample(ctx.rng)
        )
        self.gold += earn

    def _maybe_purchase_item(self, ctx: Any) -> None:
        item = str(ctx.rng.choice(_ITEM_TYPES))
        price = _market_prices[item]

        # Real money purchase (IAP)
        iap_rate = self._iap_rate[self.segment]
        if ctx.rng.random() < iap_rate:
            iap_gold = price * 5  # IAP gives 5× value
            self.gold += iap_gold
            ctx.emit(ItemPurchaseEvent(
                player_id=self.id,
                item_type=item,
                price=price,
                is_real_money=True,
            ))
            self.lifetime_spend += price
            return

        # In-game purchase
        if self.gold >= price * 0.8 and ctx.rng.random() < 0.10:
            self.gold -= price
            ctx.emit(ItemPurchaseEvent(
                player_id=self.id,
                item_type=item,
                price=price,
                is_real_money=False,
            ))

    def _maybe_level_up(self, ctx: Any) -> None:
        # Probability of leveling up decreases as level increases
        level_up_prob = max(0.01, 0.15 - self.level * 0.005)
        if ctx.rng.random() < level_up_prob:
            self.level += 1
            ctx.emit(PlayerLevelUpEvent(player_id=self.id, new_level=self.level))

    def _update_churn_risk(self, ctx: Any) -> None:
        base = 0.005
        if self.sessions > 100:
            base += 0.002  # veteran players have slight fatigue
        if self.gold < 50:
            base += 0.010  # broke players churn faster
        self.churn_risk = min(base, 0.05)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def game_economy_world(
    n_players: int = 2000,
    steps: int = 365,
    seed: int = 42,
    initial_prices: dict | None = None,
) -> Simulation:
    """
    Build a virtual game economy simulation.

    Each step represents one day of gameplay.

    Parameters
    ----------
    n_players:      Number of player agents.
    steps:          Number of simulation steps (days).
    seed:           Random seed.
    initial_prices: Override starting item market prices (dict[item_type, price]).

    Returns
    -------
    Simulation object ready to run.

    Example
    -------
    >>> sim = game_economy_world(n_players=500, steps=90)
    >>> result = sim.run()
    >>> df = result.to_pandas()["economy_metrics"]
    """
    global _market_prices
    if initial_prices:
        _market_prices.update(initial_prices)

    sim = Simulation(
        name="game_economy_world",
        seed=seed,
        clock=DiscreteClock(steps=steps),
    )
    sim.add_agents(Player, count=n_players)

    @sim.global_rule(every=7)
    def update_market_prices(ctx: Any) -> None:
        """Prices drift based on supply/demand (more purchases = price increases)."""
        recent_purchases = [
            e for e in ctx._event_log
            if isinstance(e, ItemPurchaseEvent) and not e.is_real_money
        ]
        purchase_counts: dict[str, int] = {}
        for e in recent_purchases[-200:]:   # last 200 in-game purchases
            purchase_counts[e.item_type] = purchase_counts.get(e.item_type, 0) + 1

        for item, price in _market_prices.items():
            demand = purchase_counts.get(item, 0)
            if demand > 30:
                _market_prices[item] = price * 1.05   # +5% price inflation
            elif demand < 5:
                _market_prices[item] = price * 0.97   # -3% price deflation
            # Clamp prices
            _market_prices[item] = max(5.0, min(10000.0, _market_prices[item]))

    sim.add_probe(EventLogProbe(
        events=[ItemPurchaseEvent, PlayerChurnEvent, PlayerLevelUpEvent],
        name="event_log",
    ))
    sim.add_probe(AggregatorProbe(
        metrics={
            "dau": lambda ctx: ctx.agent_count(Player),
            "iap_revenue": lambda ctx: ctx.event_sum(ItemPurchaseEvent, "price"),
            "level_ups": lambda ctx: ctx.event_count(PlayerLevelUpEvent),
            "churns": lambda ctx: ctx.event_count(PlayerChurnEvent),
            "avg_level": lambda ctx: ctx.agent_mean(Player, "level"),
            "avg_gold": lambda ctx: ctx.agent_mean(Player, "gold"),
            "sword_price": lambda ctx: _market_prices["sword"],
            "potion_price": lambda ctx: _market_prices["potion"],
        },
        every=1,
        name="economy_metrics",
    ))
    sim.add_probe(SnapshotProbe(
        agent_type=Player,
        fields=["id", "level", "gold", "segment", "lifetime_spend", "sessions"],
        every=30,
        sample_rate=0.2,
        name="player_snapshot",
    ))

    return sim
