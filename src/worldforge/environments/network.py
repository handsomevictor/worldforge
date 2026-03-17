"""NetworkEnvironment: graph-based spatial structure using NetworkX."""
from __future__ import annotations

from typing import Any

from worldforge.environments.base import Environment
from worldforge.core.exceptions import ConfigurationError


class NetworkEnvironment(Environment):
    """
    Agent interaction topology based on a graph (requires networkx).

    Agents are nodes; edges represent relationships/connections.

    Example::

        env = NetworkEnvironment.scale_free(n=1000, m=3)
        sim = Simulation(...)
        sim.environment = env

        # In agent.step():
        neighbors = ctx.environment.neighbors(self.id)
    """

    def __init__(self, graph: Any | None = None) -> None:
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for NetworkEnvironment. "
                "Install it with: pip install worldforge[network]"
            ) from e
        import networkx as nx
        self._graph: Any = graph if graph is not None else nx.Graph()
        self._agent_map: dict[str, Any] = {}  # agent_id -> agent object

    # -------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------

    @classmethod
    def scale_free(cls, n: int, m: int = 2) -> "NetworkEnvironment":
        """Barabási–Albert scale-free network."""
        import networkx as nx
        g = nx.barabasi_albert_graph(n, m)
        return cls(graph=nx.relabel_nodes(g, {i: str(i) for i in g.nodes}))

    @classmethod
    def erdos_renyi(cls, n: int, p: float) -> "NetworkEnvironment":
        """Erdős–Rényi random graph."""
        import networkx as nx
        g = nx.erdos_renyi_graph(n, p)
        return cls(graph=nx.relabel_nodes(g, {i: str(i) for i in g.nodes}))

    @classmethod
    def small_world(cls, n: int, k: int = 4, p: float = 0.1) -> "NetworkEnvironment":
        """Watts–Strogatz small-world network."""
        import networkx as nx
        g = nx.watts_strogatz_graph(n, k, p)
        return cls(graph=nx.relabel_nodes(g, {i: str(i) for i in g.nodes}))

    @classmethod
    def from_edgelist(cls, path: str, delimiter: str = ",") -> "NetworkEnvironment":
        """Load from a CSV edge list file."""
        import networkx as nx
        g = nx.read_edgelist(path, delimiter=delimiter)
        return cls(graph=g)

    # -------------------------------------------------------------------
    # Environment interface
    # -------------------------------------------------------------------

    def add_agent(self, agent: Any) -> None:
        self._agent_map[agent.id] = agent
        if agent.id not in self._graph:
            self._graph.add_node(agent.id)

    def remove_agent(self, agent: Any) -> None:
        self._agent_map.pop(agent.id, None)
        if agent.id in self._graph:
            self._graph.remove_node(agent.id)

    def agents(self) -> list:
        return list(self._agent_map.values())

    # -------------------------------------------------------------------
    # Graph queries
    # -------------------------------------------------------------------

    def neighbors(self, agent_id: str) -> list:
        """Return agent objects adjacent to the given agent."""
        return [
            self._agent_map[n]
            for n in self._graph.neighbors(agent_id)
            if n in self._agent_map
        ]

    def add_edge(self, a_id: str, b_id: str, **attrs: Any) -> None:
        self._graph.add_edge(a_id, b_id, **attrs)

    def remove_edge(self, a_id: str, b_id: str) -> None:
        if self._graph.has_edge(a_id, b_id):
            self._graph.remove_edge(a_id, b_id)

    def agents_within_hops(self, agent: Any, hops: int) -> list:
        """Return all agents reachable from `agent` within `hops` edges."""
        import networkx as nx
        if agent.id not in self._graph:
            return []
        reachable = nx.single_source_shortest_path_length(
            self._graph, agent.id, cutoff=hops
        )
        return [
            self._agent_map[nid]
            for nid in reachable
            if nid != agent.id and nid in self._agent_map
        ]

    def degree(self, agent_id: str) -> int:
        return self._graph.degree(agent_id)

    @property
    def graph(self) -> Any:
        """Direct access to the underlying networkx Graph."""
        return self._graph

    def __repr__(self) -> str:
        return (
            f"NetworkEnvironment(nodes={self._graph.number_of_nodes()}, "
            f"edges={self._graph.number_of_edges()})"
        )
