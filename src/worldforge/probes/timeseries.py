"""TimeSeriesProbe: high-frequency time series for specific metrics."""
from __future__ import annotations

from typing import Any, Callable

from worldforge.probes.base import Probe


class TimeSeriesProbe(Probe):
    """
    Records named scalar metrics at every step (or configured interval).

    Intended for high-resolution time series data (e.g., every hour).

    Example::

        sim.add_probe(TimeSeriesProbe(
            series={
                "avg_balance": lambda ctx: ctx.agent_mean(User, "balance"),
                "p95_balance": lambda ctx: ctx.agent_percentile(User, "balance", 0.95),
            },
            every="1 hour",
            name="timeseries",
        ))
    """

    def __init__(
        self,
        series: dict[str, Callable],
        every: Any = 1,
        name: str = "timeseries",
    ) -> None:
        super().__init__(every=every, name=name)
        self.series = series
        self._records: list[dict] = []

    def collect(self, ctx: Any) -> None:
        record = {"timestamp": ctx.now}
        for series_name, fn in self.series.items():
            try:
                record[series_name] = fn(ctx)
            except Exception:
                record[series_name] = None
        self._records.append(record)

    def finalize(self) -> list:
        return list(self._records)


class CustomProbe(Probe):
    """
    User-defined probe created via @sim.probe(every=...) decorator.

    The decorated function signature: fn(ctx, collector) where
    `collector.record(dict)` appends a row.
    """

    def __init__(self, fn: Callable, every: Any = 1, name: str = "") -> None:
        super().__init__(every=every, name=name or fn.__name__)
        self._fn = fn
        self._records: list[dict] = []

    def collect(self, ctx: Any) -> None:
        class _Collector:
            def __init__(self, records: list) -> None:
                self._records = records

            def record(self, row: dict) -> None:
                self._records.append(row)

        self._fn(ctx, _Collector(self._records))

    def finalize(self) -> list:
        return list(self._records)
