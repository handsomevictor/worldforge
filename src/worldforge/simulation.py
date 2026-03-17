"""Simulation: the top-level orchestrator for a worldforge simulation."""
from __future__ import annotations

from typing import Any, Callable

from worldforge.core.clock import DiscreteClock
from worldforge.output.result import SimulationResult


class Simulation:
    """
    The main entry point for building and running a worldforge simulation.

    Usage::

        sim = Simulation(name="my_sim", seed=42, clock=DiscreteClock(steps=100))
        sim.add_agents(User, count=1000)
        sim.add_probe(AggregatorProbe(...))
        result = sim.run()

    Parameters
    ----------
    name:   Human-readable simulation name (used in result metadata).
    seed:   Random seed for reproducibility.
    clock:  A Clock instance (DiscreteClock, CalendarClock, EventDrivenClock).
            Defaults to DiscreteClock(steps=100).
    """

    def __init__(
        self,
        name: str = "simulation",
        seed: int = 42,
        clock: Any = None,
    ) -> None:
        self.name = name
        self.seed = seed
        self.clock = clock if clock is not None else DiscreteClock(steps=100)

        # Agent specs: (agent_type, count, factory_fn | None)
        self._agent_specs: list[tuple[type, int, Callable | None]] = []

        # Probes
        self._probes: list[Any] = []

        # External shocks
        self._shocks: list[Any] = []

        # Event handlers: [(event_type, handler_fn), ...]
        self._event_handlers: list[tuple[type, Callable]] = []

        # Global rules: [(fn, every), ...]
        self._global_rules: list[tuple[Callable, Any]] = []

        # Optional environment (set via set_environment())
        self._environment: Any = None

    # -------------------------------------------------------------------
    # Building the simulation
    # -------------------------------------------------------------------

    def add_agents(
        self,
        agent_type: type,
        count: int = 1,
        factory: Callable | None = None,
    ) -> "Simulation":
        """
        Schedule `count` agents of `agent_type` to be added at simulation start.

        Parameters
        ----------
        agent_type: Agent subclass to instantiate.
        count:      Number of agents to add.
        factory:    Optional callable(i: int, rng) → Agent. When provided,
                    each agent is created by calling factory(i, rng) where
                    i is the index (0-based).
        """
        self._agent_specs.append((agent_type, count, factory))
        return self

    def add_probe(self, probe: Any) -> "Simulation":
        """Register a data-collection probe."""
        self._probes.append(probe)
        return self

    def add_shock(self, shock: Any) -> "Simulation":
        """Register an ExternalShock."""
        self._shocks.append(shock)
        return self

    def set_environment(self, env: Any) -> "Simulation":
        """
        Attach an environment (NetworkEnvironment, GridEnvironment, MarketEnvironment, etc.).

        The environment is made available to agents via ``ctx.environment`` during the run.

        Usage::

            env = MarketEnvironment(assets=["STOCK"], initial_prices={"STOCK": 100})
            sim.set_environment(env)
        """
        self._environment = env
        return self

    def on(self, event_type: type) -> Callable:
        """
        Decorator: register a global event handler for `event_type`.

        Usage::

            @sim.on(PurchaseEvent)
            def handle_purchase(event, ctx):
                ...
        """
        def decorator(fn: Callable) -> Callable:
            self._event_handlers.append((event_type, fn))
            return fn
        return decorator

    def global_rule(self, every: Any = 1) -> Callable:
        """
        Decorator: register a global rule called every `every` steps.

        Usage::

            @sim.global_rule(every="1 week")
            def weekly_check(ctx):
                ...
        """
        def decorator(fn: Callable) -> Callable:
            self._global_rules.append((fn, every))
            return fn
        return decorator

    def probe(self, every: Any = 1) -> Callable:
        """
        Decorator: create and register a CustomProbe from a function.

        The decorated function receives (ctx, collector). Use
        collector.record(dict) to save rows.

        Usage::

            @sim.probe(every="1 week")
            def weekly_cohort(ctx, collector):
                collector.record({"week": ctx.now, "n": ctx.agent_count(User)})
        """
        def decorator(fn: Callable) -> Callable:
            from worldforge.probes.timeseries import CustomProbe
            p = CustomProbe(fn=fn, every=every)
            self._probes.append(p)
            return fn
        return decorator

    # -------------------------------------------------------------------
    # Running
    # -------------------------------------------------------------------

    def run(self, progress: bool = False) -> SimulationResult:
        """
        Execute the simulation and return a SimulationResult.

        Parameters
        ----------
        progress: If True, print a progress bar to stdout.
        """
        from worldforge.runner.sequential import SequentialRunner
        runner = SequentialRunner(self)
        return runner.run(progress=progress)

    # -------------------------------------------------------------------
    # Checkpointing (basic pickle support)
    # -------------------------------------------------------------------

    def checkpoint(self, path: str) -> None:
        """Save simulation state to a pickle file."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_checkpoint(cls, path: str) -> "Simulation":
        """Load simulation state from a pickle file."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return (
            f"Simulation(name={self.name!r}, seed={self.seed}, "
            f"clock={self.clock!r})"
        )
