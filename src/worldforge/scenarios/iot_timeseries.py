"""iot_world: IoT sensor time series simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldforge import Agent, Simulation, field
from worldforge.distributions import Normal, Uniform, Categorical
from worldforge.events.base import Event
from worldforge.probes import EventLogProbe, TimeSeriesProbe, SnapshotProbe


@dataclass
class SensorReading(Event):
    sensor_id: str
    sensor_type: str
    value: float
    is_anomaly: bool


_SENSOR_PARAMS = {
    "temperature": {"mu": 22.0, "sigma": 1.5, "anomaly_sigma": 10.0},
    "pressure":    {"mu": 101.3, "sigma": 0.5, "anomaly_sigma": 5.0},
    "vibration":   {"mu": 0.1, "sigma": 0.02, "anomaly_sigma": 0.5},
}


class Sensor(Agent):
    sensor_type: str = field(Categorical(
        choices=["temperature", "pressure", "vibration"],
        weights=[0.5, 0.3, 0.2],
    ))
    anomaly_rate: float = field(0.005)
    _drift: float = field(0.0)

    def step(self, ctx: Any) -> None:
        params = _SENSOR_PARAMS[self.sensor_type]
        is_anomaly = ctx.rng.random() < self.anomaly_rate

        if is_anomaly:
            sigma = params["anomaly_sigma"]
        else:
            sigma = params["sigma"]

        # Small drift over time
        self._drift += float(Normal(mu=0, sigma=0.001).sample(ctx.rng))
        value = float(Normal(
            mu=params["mu"] + self._drift,
            sigma=sigma,
        ).sample(ctx.rng))

        ctx.emit(SensorReading(
            sensor_id=self.id,
            sensor_type=self.sensor_type,
            value=value,
            is_anomaly=is_anomaly,
        ))


def iot_world(
    n_sensors: int = 100,
    duration_steps: int = 1440,  # 24 hours * 60 minutes
    anomaly_rate: float = 0.005,
    seed: int = 42,
) -> Simulation:
    """
    Build an IoT sensor simulation.

    Each step represents one sample interval (e.g., 1 minute).
    """
    from worldforge.core.clock import DiscreteClock

    sim = Simulation(
        name="iot_world",
        seed=seed,
        clock=DiscreteClock(steps=duration_steps),
    )
    sim.add_agents(
        Sensor,
        count=n_sensors,
        factory=lambda i, rng: Sensor(anomaly_rate=anomaly_rate),
    )

    sim.add_probe(EventLogProbe(
        events=[SensorReading],
        name="sensor_readings",
    ))
    sim.add_probe(TimeSeriesProbe(
        series={
            "n_anomalies": lambda ctx: ctx.event_count(SensorReading),
        },
        every=60,
        name="hourly_summary",
    ))

    return sim
