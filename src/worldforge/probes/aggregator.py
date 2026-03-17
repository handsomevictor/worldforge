"""AggregatorProbe: periodic aggregate metrics computed from ctx."""
from __future__ import annotations

from typing import Any, Callable

from worldforge.probes.base import Probe


class AggregatorProbe(Probe):
    """
    Computes named aggregate metrics at regular intervals.

    Example::

        sim.add_probe(AggregatorProbe(
            metrics={
                "dau":       lambda ctx: ctx.agent_count(User),
                "gmv_daily": lambda ctx: ctx.event_sum(PurchaseEvent, "amount", last="1 day"),
                "avg_bal":   lambda ctx: ctx.agent_mean(User, "balance"),
            },
            every="1 day",
            name="aggregator",
        ))
    """

    def __init__(
        self,
        metrics: dict[str, Callable],
        every: Any = 1,
        name: str = "aggregator",
    ) -> None:
        super().__init__(every=every, name=name)
        self.metrics = metrics
        self._records: list[dict] = []

    def collect(self, ctx: Any) -> None:
        record = {"timestamp": ctx.now}
        for metric_name, fn in self.metrics.items():
            try:
                record[metric_name] = fn(ctx)
            except Exception as exc:
                record[metric_name] = None
        self._records.append(record)

    def finalize(self) -> list:
        return list(self._records)
