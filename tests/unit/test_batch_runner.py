"""Unit tests for BatchRunner and BatchResult."""
from __future__ import annotations

import pytest

from worldforge import Agent, Simulation, field
from worldforge.agent import _reset_id_counter
from worldforge.core.clock import DiscreteClock
from worldforge.distributions import Uniform
from worldforge.probes import AggregatorProbe
from worldforge.runner.batch import BatchRunner, BatchResult
from worldforge.output.result import SimulationResult


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


class SimpleAgent(Agent):
    value: float = field(0.0)

    def step(self, ctx):
        self.value += 1.0


def build_sim(n_steps=10, n_agents=5):
    """Factory that builds a simple testable simulation."""
    sim = Simulation(
        name="batch_test",
        seed=42,
        clock=DiscreteClock(steps=n_steps),
    )
    sim.add_agents(SimpleAgent, count=n_agents)
    sim.add_probe(AggregatorProbe(
        metrics={"n": lambda ctx: ctx.agent_count(SimpleAgent)},
        every=1,
        name="metrics",
    ))
    return sim


class TestBatchRunner:
    def test_single_run_sequential(self):
        batch = BatchRunner(
            sim_factory=lambda p: build_sim(n_steps=p["steps"]),
            param_grid={"steps": [5]},
            n_replications=1,
            workers=1,
        )
        br = batch.run()
        assert len(br) == 1
        assert isinstance(br.runs[0]["result"], SimulationResult)

    def test_grid_expansion(self):
        """3 step values × 2 agent counts = 6 parameter sets."""
        batch = BatchRunner(
            sim_factory=lambda p: build_sim(n_steps=p["steps"], n_agents=p["n"]),
            param_grid={
                "steps": [5, 10, 20],
                "n": [3, 5],
            },
            n_replications=1,
            workers=1,
        )
        br = batch.run()
        assert len(br) == 6

    def test_n_replications(self):
        """3 replications of a single parameter set → 3 runs."""
        batch = BatchRunner(
            sim_factory=lambda p: build_sim(n_steps=5),
            param_grid={"dummy": [1]},
            n_replications=3,
            workers=1,
        )
        br = batch.run()
        assert len(br) == 3
        reps = [r["replication"] for r in br.runs]
        assert sorted(reps) == [0, 1, 2]

    def test_distribution_sampling(self):
        """Distribution params are sampled n_samples times."""
        batch = BatchRunner(
            sim_factory=lambda p: build_sim(n_steps=5),
            param_grid={"alpha": Uniform(0.0, 1.0)},
            n_samples=4,
            n_replications=1,
            workers=1,
        )
        br = batch.run()
        # n_samples=4, 1 list_param (none), → 4 parameter sets
        assert len(br) == 4
        alphas = [r["params"]["alpha"] for r in br.runs]
        # All should be in [0, 1]
        assert all(0.0 <= a <= 1.0 for a in alphas)

    def test_params_stored_in_result(self):
        batch = BatchRunner(
            sim_factory=lambda p: build_sim(n_steps=p["steps"]),
            param_grid={"steps": [7]},
            n_replications=1,
            workers=1,
        )
        br = batch.run()
        assert br.runs[0]["params"]["steps"] == 7


class TestBatchResult:
    def _make_result(self):
        batch = BatchRunner(
            sim_factory=lambda p: build_sim(n_steps=p["steps"]),
            param_grid={"steps": [5, 10]},
            n_replications=2,
            workers=1,
        )
        return batch.run()

    def test_len(self):
        br = self._make_result()
        assert len(br) == 4  # 2 param sets × 2 replications

    def test_to_pandas(self):
        pd = pytest.importorskip("pandas")
        br = self._make_result()
        df = br.to_pandas()
        assert hasattr(df, "columns")
        assert "steps" in df.columns
        assert "replication" in df.columns
        assert len(df) == 4

    def test_to_pandas_includes_metadata(self):
        pd = pytest.importorskip("pandas")
        br = self._make_result()
        df = br.to_pandas()
        # result.metadata includes "steps", "name", "seed", etc.
        assert "name" in df.columns or "seed" in df.columns

    def test_repr(self):
        batch = BatchRunner(
            sim_factory=lambda p: build_sim(),
            param_grid={"x": [1]},
            n_replications=1,
            workers=1,
        )
        br = batch.run()
        assert "BatchResult" in repr(br)
