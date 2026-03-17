"""Performance benchmarks: verify worldforge meets stated throughput targets.

Targets from CLAUDE.md §八:
  - 1,000 agents × 1,000 steps  < 1 second
  - 100,000 agents × 365 steps  < 30 seconds   (skipped unless --run-slow)
"""
from __future__ import annotations

import time

import pytest

from worldforge import Agent, Simulation, field
from worldforge.agent import _reset_id_counter
from worldforge.core.clock import DiscreteClock
from worldforge.distributions import Normal, Categorical


# ---------------------------------------------------------------------------
# Minimal agent for benchmarking (pure Python step, minimal overhead)
# ---------------------------------------------------------------------------

class BenchAgent(Agent):
    value: float = field(Normal(mu=100, sigma=20))
    counter: int = field(0)

    def step(self, ctx):
        self.value *= 1.0001
        self.counter += 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(n_agents: int, n_steps: int) -> float:
    _reset_id_counter(1)
    sim = Simulation(
        name="bench",
        seed=0,
        clock=DiscreteClock(steps=n_steps),
    )
    sim.add_agents(BenchAgent, count=n_agents)
    t0 = time.perf_counter()
    sim.run()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmarks (pytest-benchmark)
# ---------------------------------------------------------------------------

def test_bench_1k_agents_1k_steps(benchmark):
    """Target: 1,000 agents × 1,000 steps < 1 second."""
    elapsed = benchmark(_run, n_agents=1_000, n_steps=1_000)
    # benchmark() captures timing internally; we also assert the mean is fast.
    assert benchmark.stats["mean"] < 1.0, (
        f"1k×1k took {benchmark.stats['mean']:.3f}s, expected < 1s"
    )


@pytest.mark.slow
def test_bench_100k_agents_365_steps():
    """Target: 100,000 agents × 365 steps < 30 seconds (marked slow)."""
    elapsed = _run(n_agents=100_000, n_steps=365)
    assert elapsed < 30.0, f"100k×365 took {elapsed:.1f}s, expected < 30s"


# ---------------------------------------------------------------------------
# Plain timing test (no benchmark fixture — always runs)
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_1k_agents_1k_steps_within_2s(self):
        """Generous 2-second budget for CI environments."""
        elapsed = _run(n_agents=1_000, n_steps=1_000)
        assert elapsed < 2.0, f"1k×1k took {elapsed:.3f}s"

    def test_10k_agents_100_steps_within_2s(self):
        elapsed = _run(n_agents=10_000, n_steps=100)
        assert elapsed < 2.0, f"10k×100 took {elapsed:.3f}s"
