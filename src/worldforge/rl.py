"""
GymWrapper: Gymnasium-compatible environment wrapping a worldforge Simulation.

Requires: pip install gymnasium
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class GymWrapper:
    """
    Wraps a worldforge Simulation as a Gymnasium-compatible environment.

    This allows reinforcement learning agents to interact with worldforge
    simulations using the standard `reset()` / `step(action)` interface.

    Parameters
    ----------
    sim:            A configured (but not yet run) Simulation object.
    observation:    Callable(ctx) → np.ndarray.  Extracts the observation
                    vector from the simulation context each step.
    reward:         Callable(ctx) → float.  Computes the reward signal.
    action_fn:      Callable(action, ctx) → None.  Applies the agent's action
                    to the simulation (e.g. change a parameter).
    action_space:   Gymnasium space or the string "discrete" / "continuous".
                    When a string is given a minimal placeholder space is used;
                    pass an actual ``gymnasium.spaces`` object for real training.
    max_steps:      Maximum number of steps per episode (overrides clock length).

    Example
    -------
    >>> from worldforge.rl import GymWrapper
    >>> from worldforge.scenarios import saas_world
    >>>
    >>> sim = saas_world(n_users=500, steps=365)
    >>>
    >>> env = GymWrapper(
    ...     sim=sim,
    ...     observation=lambda ctx: np.array([
    ...         ctx.agent_count() / 500,
    ...         ctx.agent_mean(type(list(ctx._agents.values())[0]), "engagement")
    ...         if ctx._agents else 0.0,
    ...     ]),
    ...     reward=lambda ctx: ctx.agent_count() / 500,
    ...     action_fn=lambda action, ctx: None,   # no-op: observe only
    ... )
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(0)
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        sim: Any,
        observation: Callable,
        reward: Callable,
        action_fn: Callable | None = None,
        action_space: Any = "discrete",
        max_steps: int | None = None,
    ) -> None:
        self._sim_template = sim
        self._observation_fn = observation
        self._reward_fn = reward
        self._action_fn = action_fn or (lambda action, ctx: None)
        self._max_steps = max_steps

        self._ctx: Any = None
        self._clock: Any = None
        self._step_count: int = 0
        self._done: bool = False

        # Lazy Gymnasium space setup
        self._action_space_spec = action_space
        self._action_space: Any = None
        self._observation_space: Any = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset to the start of a new episode.

        Returns
        -------
        observation: np.ndarray
        info:        dict
        """
        from worldforge.runner.sequential import SequentialRunner
        from worldforge.core.context import SimContext

        # Re-initialise the simulation from scratch each episode
        sim = self._sim_template
        if seed is not None:
            sim.seed = seed

        rng = np.random.default_rng(sim.seed)
        clock = sim.clock
        clock.reset()

        from worldforge.agent import _reset_id_counter
        _reset_id_counter(1)

        self._ctx = SimContext(clock=clock, rng=rng)

        # Register event handlers
        for event_type, handler in sim._event_handlers:
            self._ctx.register_event_handler(event_type, handler)

        # Add initial agents
        for agent_type, count, factory in sim._agent_specs:
            for i in range(count):
                if factory is not None:
                    agent = factory(i, rng)
                else:
                    agent = agent_type(_rng=rng)
                self._ctx._register_agent(agent)
                agent.on_born(self._ctx)

        self._step_count = 0
        self._done = False

        obs = self._observation_fn(self._ctx)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        """
        Advance the simulation by one step, applying `action`.

        Returns
        -------
        observation:  np.ndarray
        reward:       float
        terminated:   bool  (simulation reached its natural end)
        truncated:    bool  (max_steps exceeded)
        info:         dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        ctx = self._ctx
        clock = self._sim_template.clock

        # Apply action to context before ticking
        self._action_fn(action, ctx)

        # Tick the clock
        clock.tick()
        self._step_count += 1

        # Run agents
        ctx._run_tick()

        # Compute reward before checking done
        reward = float(self._reward_fn(ctx))

        obs = np.asarray(self._observation_fn(ctx), dtype=np.float32)

        terminated = clock.is_done
        truncated = (
            self._max_steps is not None and self._step_count >= self._max_steps
        )
        self._done = terminated or truncated

        return obs, reward, terminated, truncated, {}

    def render(self):
        """No rendering implemented (override in subclass if needed)."""
        pass

    def close(self):
        """Clean up resources."""
        self._ctx = None

    # ------------------------------------------------------------------
    # Spaces (lazy, minimal stubs — replace with real gym spaces)
    # ------------------------------------------------------------------

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = self._build_space(self._action_space_spec)
        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            # Derive shape from a dummy observation (requires reset first)
            try:
                gym = __import__("gymnasium")
                obs, _ = self.reset()
                self._observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
                )
            except Exception:
                self._observation_space = None
        return self._observation_space

    @staticmethod
    def _build_space(spec: Any) -> Any:
        try:
            gym = __import__("gymnasium")
        except ImportError:
            return spec   # return as-is if gymnasium not installed

        if isinstance(spec, str):
            if spec == "discrete":
                return gym.spaces.Discrete(2)
            if spec == "continuous":
                return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        return spec   # already a gymnasium space
