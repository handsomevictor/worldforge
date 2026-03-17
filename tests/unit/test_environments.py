"""Unit tests for environments: GridEnvironment, ContinuousSpace, MarketEnvironment.
NetworkEnvironment tests require networkx (optional dep) and are skipped if absent."""
from __future__ import annotations

import math
import pytest

from worldforge.agent import Agent, _reset_id_counter
from worldforge.core.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


def _make_agent(id_override=None):
    a = Agent()
    if id_override:
        object.__setattr__(a, 'id', id_override)
    return a


# ============================================================
# GridEnvironment
# ============================================================

class TestGridEnvironment:
    from worldforge.environments.grid import GridEnvironment

    def test_place_and_position(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10)
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 3, 4)
        assert env.position(a) == (3, 4)

    def test_move(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10)
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 2, 2)
        env.move(a, 1, -1)
        assert env.position(a) == (3, 1)

    def test_bounded_clamp(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=5, height=5, topology="bounded")
        a = _make_agent()
        env.add_agent(a)
        with pytest.raises(ConfigurationError):
            env.place(a, 10, 10)

    def test_torus_wraps(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=5, height=5, topology="torus")
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 6, 7)
        assert env.position(a) == (1, 2)

    def test_moore_neighbors(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10, topology="bounded", neighborhood="moore")
        a = _make_agent()
        b = _make_agent()
        c = _make_agent()
        for ag in [a, b, c]:
            env.add_agent(ag)
        env.place(a, 5, 5)
        env.place(b, 5, 6)  # adjacent
        env.place(c, 5, 9)  # far away
        neighbors = env.neighbors(a)
        assert b in neighbors
        assert c not in neighbors
        assert a not in neighbors

    def test_von_neumann_excludes_diagonal(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10, topology="bounded", neighborhood="von_neumann")
        a = _make_agent()
        b = _make_agent()  # diagonal — should not be a von Neumann neighbor
        for ag in [a, b]:
            env.add_agent(ag)
        env.place(a, 5, 5)
        env.place(b, 6, 6)  # diagonal
        neighbors = env.neighbors(a, radius=1)
        assert b not in neighbors

    def test_agents_at(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10)
        a = _make_agent()
        b = _make_agent()
        env.add_agent(a)
        env.add_agent(b)
        env.place(a, 3, 3)
        env.place(b, 3, 3)
        at = env.agents_at(3, 3)
        assert a in at and b in at

    def test_remove_agent(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10)
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 2, 2)
        env.remove_agent(a)
        assert env.position(a) is None
        assert a not in env.agents()

    def test_no_position_before_place(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10)
        a = _make_agent()
        env.add_agent(a)
        assert env.position(a) is None

    def test_move_without_place_raises(self):
        from worldforge.environments.grid import GridEnvironment
        env = GridEnvironment(width=10, height=10)
        a = _make_agent()
        env.add_agent(a)
        with pytest.raises(ConfigurationError):
            env.move(a, 1, 0)


# ============================================================
# ContinuousSpace
# ============================================================

class TestContinuousSpace:
    def test_place_and_position(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(100.0, 100.0)
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 10.5, 20.3)
        pos = env.position(a)
        assert abs(pos[0] - 10.5) < 1e-9
        assert abs(pos[1] - 20.3) < 1e-9

    def test_move(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(100.0, 100.0)
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 10.0, 10.0)
        env.move(a, 3.5, -2.0)
        pos = env.position(a)
        assert abs(pos[0] - 13.5) < 1e-9
        assert abs(pos[1] - 8.0) < 1e-9

    def test_bounded_clamps_to_border(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(50.0, 50.0, topology="bounded")
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 200.0, -10.0)
        pos = env.position(a)
        assert pos[0] == 50.0
        assert pos[1] == 0.0

    def test_torus_wraps(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(100.0, 100.0, topology="torus")
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 110.0, -5.0)
        pos = env.position(a)
        assert abs(pos[0] - 10.0) < 1e-9
        assert abs(pos[1] - 95.0) < 1e-9

    def test_agents_near(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(100.0, 100.0)
        a = _make_agent()
        b = _make_agent()
        c = _make_agent()
        for ag in [a, b, c]:
            env.add_agent(ag)
        env.place(a, 0.0, 0.0)
        env.place(b, 3.0, 4.0)   # distance = 5
        env.place(c, 50.0, 50.0) # far away
        near = env.agents_near(a, radius=6.0)
        assert b in near
        assert c not in near
        assert a not in near

    def test_agents_near_type_filter(self):
        from worldforge.environments.continuous import ContinuousSpace

        class TypeA(Agent): pass
        class TypeB(Agent): pass

        env = ContinuousSpace(100.0, 100.0)
        ta = TypeA()
        tb = TypeB()
        origin = _make_agent()
        for ag in [origin, ta, tb]:
            env.add_agent(ag)
        env.place(origin, 0.0, 0.0)
        env.place(ta, 1.0, 0.0)
        env.place(tb, 1.0, 0.0)

        near_a = env.agents_near(origin, radius=5.0, agent_type=TypeA)
        assert ta in near_a
        assert tb not in near_a

    def test_distance(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(100.0, 100.0)
        a = _make_agent()
        b = _make_agent()
        env.add_agent(a)
        env.add_agent(b)
        env.place(a, 0.0, 0.0)
        env.place(b, 3.0, 4.0)
        assert abs(env.distance(a, b) - 5.0) < 1e-9

    def test_torus_distance_shorter_path(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(10.0, 10.0, topology="torus")
        a = _make_agent()
        b = _make_agent()
        env.add_agent(a)
        env.add_agent(b)
        env.place(a, 0.5, 0.0)
        env.place(b, 9.5, 0.0)
        # Direct: 9.0; through torus: 1.0
        assert abs(env.distance(a, b) - 1.0) < 1e-9

    def test_remove_agent(self):
        from worldforge.environments.continuous import ContinuousSpace
        env = ContinuousSpace(100.0, 100.0)
        a = _make_agent()
        env.add_agent(a)
        env.place(a, 5.0, 5.0)
        env.remove_agent(a)
        assert env.position(a) is None
        assert a not in env.agents()


# ============================================================
# MarketEnvironment + OrderBook
# ============================================================

class TestMarketEnvironment:
    def test_initial_mid_price(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(
            assets=["BTC"],
            initial_prices={"BTC": 50000.0},
        )
        assert env.mid_price("BTC") == 50000.0

    def test_buy_order_no_asks(self):
        """Buy order with no opposing asks → rests on book."""
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        trades = env.submit_order("agent1", asset="X", side="buy", price=99.0, qty=1.0)
        assert trades == []
        assert env.best_bid("X") == 99.0

    def test_sell_order_matches_existing_bid(self):
        """Sell order at <= bid price → trade executes."""
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        env.submit_order("buyer", asset="X", side="buy", price=101.0, qty=5.0)
        trades = env.submit_order("seller", asset="X", side="sell", price=100.0, qty=3.0)
        assert len(trades) == 1
        assert trades[0].qty == 3.0
        assert trades[0].buyer_id == "buyer"
        assert trades[0].seller_id == "seller"

    def test_buy_order_matches_existing_ask(self):
        """Buy order at >= ask price → trade executes."""
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        env.submit_order("seller", asset="X", side="sell", price=99.0, qty=5.0)
        trades = env.submit_order("buyer", asset="X", side="buy", price=100.0, qty=2.0)
        assert len(trades) == 1
        assert trades[0].qty == 2.0

    def test_partial_fill(self):
        """Order qty > available on book → partial fill, remainder rests."""
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        env.submit_order("seller", asset="X", side="sell", price=100.0, qty=3.0)
        trades = env.submit_order("buyer", asset="X", side="buy", price=101.0, qty=10.0)
        assert len(trades) == 1
        assert trades[0].qty == 3.0

    def test_mid_price_updates_after_trade(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        env.submit_order("seller", asset="X", side="sell", price=95.0, qty=1.0)
        env.submit_order("buyer", asset="X", side="buy", price=96.0, qty=1.0)
        # After trade at 95.0, mid should reflect 95
        assert env.mid_price("X") == 95.0

    def test_trade_history(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        env.submit_order("s", asset="X", side="sell", price=100.0, qty=1.0)
        env.submit_order("b", asset="X", side="buy", price=100.0, qty=1.0)
        history = env.trade_history("X")
        assert len(history) == 1

    def test_trade_history_last(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        for i in range(5):
            env.submit_order("s", asset="X", side="sell", price=float(99 + i), qty=1.0)
            env.submit_order("b", asset="X", side="buy", price=float(100 + i), qty=1.0)
        history = env.trade_history("X", last=3)
        assert len(history) == 3

    def test_set_price(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 100.0})
        env.set_price("X", 200.0)
        assert env.mid_price("X") == 200.0

    def test_assets_list(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(
            assets=["BTC", "ETH"],
            initial_prices={"BTC": 50000.0, "ETH": 3000.0},
        )
        assert sorted(env.assets()) == ["BTC", "ETH"]

    def test_unknown_asset_raises(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 1.0})
        with pytest.raises(KeyError):
            env.submit_order("a", asset="UNKNOWN", side="buy", price=1.0, qty=1.0)

    def test_add_remove_agent(self):
        from worldforge.environments.market import MarketEnvironment
        env = MarketEnvironment(assets=["X"], initial_prices={"X": 1.0})
        a = _make_agent()
        env.add_agent(a)
        assert a in env.agents()
        env.remove_agent(a)
        assert a not in env.agents()


# ============================================================
# NetworkEnvironment (requires networkx)
# ============================================================

networkx = pytest.importorskip("networkx", reason="networkx not installed")


class TestNetworkEnvironment:
    def test_scale_free_construction(self):
        from worldforge.environments.network import NetworkEnvironment
        env = NetworkEnvironment.scale_free(n=50, m=2)
        assert env.graph.number_of_nodes() == 50
        assert env.graph.number_of_edges() > 0

    def test_small_world_construction(self):
        from worldforge.environments.network import NetworkEnvironment
        env = NetworkEnvironment.small_world(n=30, k=4, p=0.1)
        assert env.graph.number_of_nodes() == 30

    def test_erdos_renyi_construction(self):
        from worldforge.environments.network import NetworkEnvironment
        env = NetworkEnvironment.erdos_renyi(n=20, p=0.5)
        assert env.graph.number_of_nodes() == 20

    def test_add_remove_agent(self):
        from worldforge.environments.network import NetworkEnvironment
        env = NetworkEnvironment()
        a = _make_agent()
        env.add_agent(a)
        assert a in env.agents()
        env.remove_agent(a)
        assert a not in env.agents()

    def test_add_edge_and_neighbors(self):
        from worldforge.environments.network import NetworkEnvironment
        env = NetworkEnvironment()
        a = _make_agent()
        b = _make_agent()
        env.add_agent(a)
        env.add_agent(b)
        env.add_edge(a.id, b.id)
        assert b in env.neighbors(a.id)
        assert a in env.neighbors(b.id)

    def test_remove_edge(self):
        from worldforge.environments.network import NetworkEnvironment
        env = NetworkEnvironment()
        a = _make_agent()
        b = _make_agent()
        env.add_agent(a)
        env.add_agent(b)
        env.add_edge(a.id, b.id)
        env.remove_edge(a.id, b.id)
        assert b not in env.neighbors(a.id)

    def test_agents_within_hops(self):
        import networkx as nx
        from worldforge.environments.network import NetworkEnvironment
        g = nx.path_graph(5)
        # Nodes: 0-1-2-3-4
        g = nx.relabel_nodes(g, {i: str(i) for i in g.nodes})
        env = NetworkEnvironment(graph=g)
        agents = [_make_agent(id_override=str(i)) for i in range(5)]
        for ag in agents:
            env._agent_map[ag.id] = ag
        # From node "0", within 2 hops: "1", "2"
        nearby = env.agents_within_hops(agents[0], hops=2)
        nearby_ids = {a.id for a in nearby}
        assert "1" in nearby_ids
        assert "2" in nearby_ids
        assert "3" not in nearby_ids

    def test_degree(self):
        from worldforge.environments.network import NetworkEnvironment
        env = NetworkEnvironment()
        a = _make_agent()
        b = _make_agent()
        c = _make_agent()
        for ag in [a, b, c]:
            env.add_agent(ag)
        env.add_edge(a.id, b.id)
        env.add_edge(a.id, c.id)
        assert env.degree(a.id) == 2
        assert env.degree(b.id) == 1
