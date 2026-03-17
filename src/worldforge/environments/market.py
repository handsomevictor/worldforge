"""MarketEnvironment: limit order book for market microstructure simulation."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from worldforge.environments.base import Environment


@dataclass(order=True)
class _Order:
    """Internal order representation."""
    # Sort key fields (order=True compares these in order)
    price: float
    seq: int
    # Non-sort fields
    agent_id: str = field(compare=False)
    asset: str = field(compare=False)
    side: str = field(compare=False)      # 'buy' or 'sell'
    qty: float = field(compare=False)
    order_id: int = field(compare=False)


@dataclass
class Trade:
    """A completed trade."""
    asset: str
    buyer_id: str
    seller_id: str
    price: float
    qty: float
    timestamp: Any = None


class OrderBook:
    """Single-asset limit order book."""

    def __init__(self, asset: str, initial_price: float, tick_size: float) -> None:
        self.asset = asset
        self.tick_size = tick_size
        self._mid_price = initial_price
        # bids: max-heap (negate price for min-heap trick)
        self._bids: list[_Order] = []
        # asks: min-heap
        self._asks: list[_Order] = []
        self._seq: int = 0
        self._order_counter: int = 0
        self._trade_log: list[Trade] = []

    def submit(
        self,
        agent_id: str,
        side: str,
        price: float,
        qty: float,
    ) -> list[Trade]:
        """Submit a limit order. Returns list of resulting trades."""
        self._order_counter += 1
        self._seq += 1
        order = _Order(
            price=price,
            seq=self._seq,
            agent_id=agent_id,
            asset=self.asset,
            side=side,
            qty=qty,
            order_id=self._order_counter,
        )
        trades = self._match(order)
        return trades

    def _match(self, order: _Order) -> list[Trade]:
        trades = []
        if order.side == "buy":
            # Match against asks (ascending price)
            while self._asks and order.qty > 0:
                best_ask = self._asks[0]
                if order.price < best_ask.price:
                    break
                trade_qty = min(order.qty, best_ask.qty)
                trade = Trade(
                    asset=self.asset,
                    buyer_id=order.agent_id,
                    seller_id=best_ask.agent_id,
                    price=best_ask.price,
                    qty=trade_qty,
                )
                trades.append(trade)
                self._trade_log.append(trade)
                self._mid_price = best_ask.price
                order.qty -= trade_qty
                best_ask.qty -= trade_qty
                if best_ask.qty <= 0:
                    heapq.heappop(self._asks)
            if order.qty > 0:
                # Use negative price so higher bids sort first
                neg_order = _Order(
                    price=-order.price,
                    seq=order.seq,
                    agent_id=order.agent_id,
                    asset=order.asset,
                    side=order.side,
                    qty=order.qty,
                    order_id=order.order_id,
                )
                heapq.heappush(self._bids, neg_order)
        else:
            # Sell: match against bids (highest price first — stored as negated)
            while self._bids and order.qty > 0:
                best_bid = self._bids[0]
                bid_price = -best_bid.price
                if order.price > bid_price:
                    break
                trade_qty = min(order.qty, best_bid.qty)
                trade = Trade(
                    asset=self.asset,
                    buyer_id=best_bid.agent_id,
                    seller_id=order.agent_id,
                    price=bid_price,
                    qty=trade_qty,
                )
                trades.append(trade)
                self._trade_log.append(trade)
                self._mid_price = bid_price
                order.qty -= trade_qty
                best_bid.qty -= trade_qty
                if best_bid.qty <= 0:
                    heapq.heappop(self._bids)
            if order.qty > 0:
                heapq.heappush(self._asks, order)
        return trades

    @property
    def mid_price(self) -> float:
        if self._bids and self._asks:
            return (-self._bids[0].price + self._asks[0].price) / 2
        return self._mid_price

    @property
    def best_bid(self) -> float | None:
        return -self._bids[0].price if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return self._asks[0].price if self._asks else None

    def trade_history(self, last: int | None = None) -> list[Trade]:
        if last is None:
            return list(self._trade_log)
        return list(self._trade_log[-last:])


class MarketEnvironment(Environment):
    """
    Multi-asset market environment with per-asset limit order books.

    Example::

        env = MarketEnvironment(
            assets=["BTC", "ETH"],
            initial_prices={"BTC": 50000, "ETH": 3000},
        )
        env.submit_order(agent_id, asset="BTC", side="buy", price=49900, qty=0.1)
        price = env.mid_price("BTC")
    """

    def __init__(
        self,
        assets: list[str],
        initial_prices: dict[str, float],
        tick_size: float = 0.01,
    ) -> None:
        self._books: dict[str, OrderBook] = {}
        for asset in assets:
            price = initial_prices.get(asset, 100.0)
            self._books[asset] = OrderBook(asset, price, tick_size)
        self._agents: dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Environment interface
    # -------------------------------------------------------------------

    def add_agent(self, agent: Any) -> None:
        self._agents[agent.id] = agent

    def remove_agent(self, agent: Any) -> None:
        self._agents.pop(agent.id, None)

    def agents(self) -> list:
        return list(self._agents.values())

    # -------------------------------------------------------------------
    # Market operations
    # -------------------------------------------------------------------

    def submit_order(
        self,
        agent_id: str,
        asset: str,
        side: str,
        price: float,
        qty: float,
    ) -> list[Trade]:
        """Submit a limit order. Returns resulting trades."""
        if asset not in self._books:
            raise KeyError(f"Unknown asset: {asset!r}")
        return self._books[asset].submit(agent_id, side, price, qty)

    def mid_price(self, asset: str) -> float:
        return self._books[asset].mid_price

    def best_bid(self, asset: str) -> float | None:
        return self._books[asset].best_bid

    def best_ask(self, asset: str) -> float | None:
        return self._books[asset].best_ask

    def set_price(self, asset: str, price: float) -> None:
        """Force-set the mid price (for external price injection)."""
        self._books[asset]._mid_price = price

    def trade_history(self, asset: str, last: int | None = None) -> list[Trade]:
        return self._books[asset].trade_history(last)

    def assets(self) -> list[str]:
        return list(self._books.keys())

    def __repr__(self) -> str:
        prices = {a: f"{self._books[a].mid_price:.2f}" for a in self._books}
        return f"MarketEnvironment(assets={prices})"
