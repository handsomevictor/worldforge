"""ParallelRunner: run multiple independent simulations concurrently."""
from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any, Callable

from worldforge.output.result import SimulationResult

if TYPE_CHECKING:
    from worldforge.simulation import Simulation


def _run_one(sim: "Simulation") -> SimulationResult:
    """Top-level function for pickling in ProcessPoolExecutor."""
    return sim.run()


class ParallelRunner:
    """
    Runs multiple Simulation instances concurrently using a process pool.

    Useful for Monte Carlo analysis: create N identical (or varied) simulations
    and run them in parallel.

    Example::

        runner = ParallelRunner(
            sims=[build_sim(seed=i) for i in range(20)],
            workers=4,
        )
        results = runner.run()
    """

    def __init__(
        self,
        sims: list["Simulation"],
        workers: int = 4,
        use_threads: bool = False,
    ) -> None:
        self._sims = sims
        self._workers = workers
        self._use_threads = use_threads

    def run(self) -> list[SimulationResult]:
        """Run all simulations and return results in input order."""
        executor_cls = (
            concurrent.futures.ThreadPoolExecutor
            if self._use_threads
            else concurrent.futures.ProcessPoolExecutor
        )
        with executor_cls(max_workers=self._workers) as executor:
            futures = [executor.submit(_run_one, sim) for sim in self._sims]
            results = [f.result() for f in futures]
        return results
