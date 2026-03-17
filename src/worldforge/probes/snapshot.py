"""SnapshotProbe: periodic point-in-time snapshot of agent fields."""
from __future__ import annotations

from typing import Any

from worldforge.probes.base import Probe


class SnapshotProbe(Probe):
    """
    Captures a snapshot of specified agent fields at regular intervals.

    Example::

        sim.add_probe(SnapshotProbe(
            agent_type=User,
            fields=["id", "balance", "tier"],
            every="1 week",
            sample_rate=0.10,
            name="user_snapshot",
        ))
    """

    def __init__(
        self,
        agent_type: type,
        fields: list[str],
        every: Any = 1,
        sample_rate: float = 1.0,
        name: str = "",
    ) -> None:
        super().__init__(every=every, name=name or "snapshot")
        self.agent_type = agent_type
        self.fields = fields
        self.sample_rate = sample_rate
        self._records: list[dict] = []

    def collect(self, ctx: Any) -> None:
        agents = ctx.agents(self.agent_type)
        if self.sample_rate < 1.0:
            k = max(1, int(len(agents) * self.sample_rate))
            # Use ctx.rng for reproducibility
            indices = ctx.rng.choice(len(agents), size=k, replace=False)
            agents = [agents[int(i)] for i in sorted(indices)]

        for agent in agents:
            record = {"timestamp": ctx.now}
            for field_name in self.fields:
                record[field_name] = getattr(agent, field_name, None)
            self._records.append(record)

    def finalize(self) -> list:
        return list(self._records)
