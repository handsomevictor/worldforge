"""Integration tests for built-in scenario factory functions (Steps 25-27)."""
from __future__ import annotations

import pytest

from worldforge.agent import _reset_id_counter
from worldforge.output.result import SimulationResult


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


class TestEcommerceWorld:
    def test_runs_and_returns_result(self):
        from worldforge.scenarios import ecommerce_world
        sim = ecommerce_world(n_users=20, duration="7 days", seed=1)
        result = sim.run()
        assert isinstance(result, SimulationResult)
        assert "event_log" in result
        assert "daily_metrics" in result
        assert "user_snapshot" in result

    def test_daily_metrics_row_count(self):
        from worldforge.scenarios import ecommerce_world
        sim = ecommerce_world(n_users=10, duration="5 days", seed=2)
        result = sim.run()
        # CalendarClock 5 days → 5 steps, every="1 day"
        rows = result["daily_metrics"]
        assert len(rows) == 5

    def test_reproducible(self):
        from worldforge.scenarios import ecommerce_world
        r1 = ecommerce_world(n_users=10, duration="3 days", seed=42).run()
        _reset_id_counter(1)
        r2 = ecommerce_world(n_users=10, duration="3 days", seed=42).run()
        assert r1["daily_metrics"] == r2["daily_metrics"]


class TestEpidemicWorld:
    def test_runs(self):
        from worldforge.scenarios import epidemic_world
        sim = epidemic_world(population=100, duration_days=30, seed=7)
        result = sim.run()
        assert "sir_curve" in result
        rows = result["sir_curve"]
        assert len(rows) == 30

    def test_infected_eventually_nonzero(self):
        from worldforge.scenarios import epidemic_world
        sim = epidemic_world(
            population=500,
            initial_infected=20,
            transmission_prob=0.5,
            duration_days=20,
            seed=42,
        )
        result = sim.run()
        rows = result["sir_curve"]
        max_infected = max(r["I"] for r in rows)
        assert max_infected > 0


class TestFintechWorld:
    def test_runs(self):
        from worldforge.scenarios import fintech_world
        result = fintech_world(n_users=50, duration_days=60, seed=3).run()
        assert "event_log" in result
        assert "monthly_metrics" in result


class TestSaasWorld:
    def test_runs(self):
        from worldforge.scenarios import saas_world
        result = saas_world(n_users=100, duration_days=60, seed=5).run()
        assert "monthly_metrics" in result
        rows = result["monthly_metrics"]
        assert len(rows) == 2  # every=30, 60 steps → 2 records

    def test_mrr_is_nonnegative(self):
        from worldforge.scenarios import saas_world
        result = saas_world(n_users=50, duration_days=30, seed=6).run()
        for row in result["monthly_metrics"]:
            assert row["mrr"] >= 0


class TestIoTWorld:
    def test_runs(self):
        from worldforge.scenarios import iot_world
        result = iot_world(n_sensors=20, duration_steps=60, seed=8).run()
        assert "sensor_readings" in result

    def test_reading_fields(self):
        from worldforge.scenarios import iot_world
        result = iot_world(n_sensors=5, duration_steps=10, seed=9).run()
        for row in result["sensor_readings"]:
            assert "sensor_id" in row
            assert "value" in row
            assert "is_anomaly" in row


class TestSupplyChainWorld:
    def test_runs(self):
        from worldforge.scenarios import supply_chain_world
        result = supply_chain_world(n_retailers=10, duration_days=30, seed=10).run()
        assert "weekly_metrics" in result

    def test_avg_inventory_nonnegative(self):
        from worldforge.scenarios import supply_chain_world
        result = supply_chain_world(n_retailers=5, duration_days=14, seed=11).run()
        for row in result["weekly_metrics"]:
            assert row["avg_inventory"] >= 0


class TestSocialNetworkWorld:
    def test_runs(self):
        from worldforge.scenarios import social_network_world
        result = social_network_world(n_users=50, duration_steps=20, seed=12).run()
        assert "opinion_timeseries" in result
        rows = result["opinion_timeseries"]
        assert len(rows) == 20
