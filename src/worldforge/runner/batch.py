"""BatchRunner: Monte Carlo parameter sweep over a simulation factory."""
from __future__ import annotations

import itertools
from typing import Any, Callable

from worldforge.output.result import SimulationResult
from worldforge.distributions.base import Distribution


class BatchResult:
    """
    Container for batch (Monte Carlo) simulation results.

    Each entry in `runs` is a dict with:
    - 'params': the parameter dict used for this run
    - 'replication': replication index (0-based)
    - 'result': SimulationResult object
    """

    def __init__(self, runs: list[dict]) -> None:
        self.runs = runs

    def to_pandas(self) -> Any:
        """
        Flatten metadata from each run's result into a pandas DataFrame.
        One row per run. Each run's result metadata fields are columns.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas required for BatchResult.to_pandas()") from e

        rows = []
        for entry in self.runs:
            row = dict(entry["params"])
            row["replication"] = entry["replication"]
            row.update(entry["result"].metadata)
            rows.append(row)
        return pd.DataFrame(rows)

    def sensitivity_analysis(self, metric: str) -> Any:
        """
        Basic one-at-a-time sensitivity: return a DataFrame showing how
        `metric` (from result metadata) varies with each parameter.
        Requires pandas.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas required for sensitivity_analysis()") from e

        rows = []
        for entry in self.runs:
            row = dict(entry["params"])
            row["replication"] = entry["replication"]
            row[metric] = entry["result"].metadata.get(metric)
            rows.append(row)
        df = pd.DataFrame(rows)
        param_cols = [c for c in df.columns if c not in ("replication", metric)]
        print(f"\nSensitivity analysis for '{metric}':")
        for col in param_cols:
            grouped = df.groupby(col)[metric].mean()
            print(f"  {col}:\n{grouped.to_string()}\n")
        return df

    def __len__(self) -> int:
        return len(self.runs)

    def __repr__(self) -> str:
        return f"BatchResult(n_runs={len(self.runs)})"


class BatchRunner:
    """
    Sweep a simulation factory over a parameter grid (Monte Carlo).

    Parameters
    ----------
    sim_factory:     callable(params: dict) -> Simulation
    param_grid:      dict of param_name -> list of values OR a Distribution
                     (sampled n_samples times)
    n_samples:       if any param is a Distribution, sample this many combos
    n_replications:  number of independent replications per parameter set
    workers:         number of parallel processes (1 = sequential)

    Example::

        batch = BatchRunner(
            sim_factory=lambda p: build_sim(**p),
            param_grid={
                "churn_rate": [0.01, 0.05, 0.10],
                "n_users":    [1000, 5000],
            },
            n_replications=3,
            workers=4,
        )
        batch_result = batch.run()
        df = batch_result.to_pandas()
    """

    def __init__(
        self,
        sim_factory: Callable[[dict], Any],
        param_grid: dict[str, Any],
        n_samples: int = 10,
        n_replications: int = 1,
        workers: int = 1,
    ) -> None:
        self._factory = sim_factory
        self._param_grid = param_grid
        self._n_samples = n_samples
        self._n_replications = n_replications
        self._workers = workers

    def _build_param_sets(self) -> list[dict]:
        """Expand param_grid into a flat list of parameter dicts."""
        import numpy as np
        rng = np.random.default_rng(0)

        list_params: dict[str, list] = {}
        dist_params: dict[str, Distribution] = {}

        for k, v in self._param_grid.items():
            if isinstance(v, Distribution):
                dist_params[k] = v
            else:
                list_params[k] = list(v)

        # Grid from list params
        if list_params:
            keys = list(list_params.keys())
            combos = list(itertools.product(*[list_params[k] for k in keys]))
            grid_sets = [dict(zip(keys, c)) for c in combos]
        else:
            grid_sets = [{}]

        if not dist_params:
            return grid_sets

        # Sample dist params and combine with grid
        result = []
        for _ in range(self._n_samples):
            sampled = {k: d.sample(rng) for k, d in dist_params.items()}
            for gs in grid_sets:
                result.append({**gs, **sampled})
        return result

    def run(self) -> BatchResult:
        """Execute all parameter combinations and return a BatchResult."""
        param_sets = self._build_param_sets()
        runs_input: list[tuple[dict, int]] = []
        for params in param_sets:
            for rep in range(self._n_replications):
                runs_input.append((params, rep))

        if self._workers <= 1:
            run_results = []
            for params, rep in runs_input:
                sim = self._factory(params)
                result = sim.run()
                run_results.append({"params": params, "replication": rep, "result": result})
        else:
            import concurrent.futures

            def _run(args: tuple) -> dict:
                params, rep = args
                sim = self._factory(params)
                result = sim.run()
                return {"params": params, "replication": rep, "result": result}

            with concurrent.futures.ProcessPoolExecutor(max_workers=self._workers) as ex:
                run_results = list(ex.map(_run, runs_input))

        return BatchResult(runs=run_results)
