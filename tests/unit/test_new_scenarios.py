"""Unit tests for the 4 new built-in scenarios."""
from __future__ import annotations

import pytest

from worldforge.agent import _reset_id_counter


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


# ---------------------------------------------------------------------------
# Rideshare
# ---------------------------------------------------------------------------

class TestRideshareWorld:
    def test_runs_and_returns_result(self):
        from worldforge.scenarios.rideshare import rideshare_world
        sim = rideshare_world(n_drivers=20, n_riders=50, steps=10, seed=42)
        result = sim.run()
        assert "platform_metrics" in result
        assert "event_log" in result

    def test_metrics_shape(self):
        from worldforge.scenarios.rideshare import rideshare_world
        sim = rideshare_world(n_drivers=10, n_riders=30, steps=5, seed=1)
        result = sim.run()
        rows = result["platform_metrics"]
        assert len(rows) == 5   # one row per step
        assert "rides_completed" in rows[0]
        assert "surge_multiplier" in rows[0]

    def test_completed_rides_nonnegative(self):
        from worldforge.scenarios.rideshare import rideshare_world
        sim = rideshare_world(n_drivers=50, n_riders=100, steps=20, seed=7)
        result = sim.run()
        for row in result["platform_metrics"]:
            assert row["rides_completed"] >= 0
            assert row["surge_multiplier"] >= 1.0

    def test_gmv_nonnegative(self):
        from worldforge.scenarios.rideshare import rideshare_world
        sim = rideshare_world(n_drivers=20, n_riders=80, steps=15, seed=3)
        result = sim.run()
        for row in result["platform_metrics"]:
            assert row["gmv"] >= 0.0

    def test_event_log_contains_ride_events(self):
        from worldforge.scenarios.rideshare import rideshare_world, RideCompletedEvent
        sim = rideshare_world(n_drivers=30, n_riders=100, steps=20, seed=42)
        result = sim.run()
        events = result["event_log"]
        assert any(e["event_type"] == "RideCompletedEvent" for e in events)


# ---------------------------------------------------------------------------
# Game Economy
# ---------------------------------------------------------------------------

class TestGameEconomyWorld:
    def test_runs_and_returns_result(self):
        from worldforge.scenarios.game_economy import game_economy_world
        sim = game_economy_world(n_players=100, steps=10, seed=42)
        result = sim.run()
        assert "economy_metrics" in result
        assert "event_log" in result

    def test_dau_decreasing_over_time(self):
        """Some players churn so DAU should generally decline."""
        from worldforge.scenarios.game_economy import game_economy_world
        sim = game_economy_world(n_players=500, steps=30, seed=42)
        result = sim.run()
        rows = result["economy_metrics"]
        assert rows[0]["dau"] >= rows[-1]["dau"]

    def test_item_purchase_events_present(self):
        from worldforge.scenarios.game_economy import game_economy_world
        sim = game_economy_world(n_players=200, steps=10, seed=42)
        result = sim.run()
        events = result["event_log"]
        assert any(e["event_type"] == "ItemPurchaseEvent" for e in events)

    def test_metrics_include_prices(self):
        from worldforge.scenarios.game_economy import game_economy_world
        sim = game_economy_world(n_players=50, steps=5, seed=1)
        result = sim.run()
        rows = result["economy_metrics"]
        assert "sword_price" in rows[0]
        assert "potion_price" in rows[0]
        assert rows[0]["sword_price"] > 0

    def test_player_snapshot_sampled(self):
        from worldforge.scenarios.game_economy import game_economy_world
        sim = game_economy_world(n_players=200, steps=30, seed=42)
        result = sim.run()
        snapshots = result["player_snapshot"]
        assert len(snapshots) > 0
        assert "level" in snapshots[0]
        assert "gold" in snapshots[0]

    def test_custom_initial_prices(self):
        from worldforge.scenarios.game_economy import game_economy_world, _market_prices
        sim = game_economy_world(
            n_players=50, steps=5, seed=1,
            initial_prices={"sword": 9999.0},
        )
        result = sim.run()
        rows = result["economy_metrics"]
        # Sword price should start near 9999 (may drift slightly after 7-step rule)
        assert rows[0]["sword_price"] >= 9000


# ---------------------------------------------------------------------------
# Org Dynamics
# ---------------------------------------------------------------------------

class TestOrgDynamicsWorld:
    def test_runs_and_returns_result(self):
        from worldforge.scenarios.org_dynamics import org_dynamics_world
        sim = org_dynamics_world(n_employees=100, steps=6, seed=42)
        result = sim.run()
        assert "org_metrics" in result
        assert "event_log" in result

    def test_headcount_tracked(self):
        from worldforge.scenarios.org_dynamics import org_dynamics_world
        sim = org_dynamics_world(n_employees=100, steps=6, seed=42)
        result = sim.run()
        rows = result["org_metrics"]
        assert len(rows) == 6
        assert "headcount" in rows[0]
        assert rows[0]["headcount"] > 0

    def test_attrition_events_present_after_enough_steps(self):
        """With enough steps, some employees churn."""
        from worldforge.scenarios.org_dynamics import org_dynamics_world
        sim = org_dynamics_world(n_employees=500, steps=12, seed=42)
        result = sim.run()
        events = result["event_log"]
        assert any(e["event_type"] == "AttritionEvent" for e in events)

    def test_avg_salary_positive(self):
        from worldforge.scenarios.org_dynamics import org_dynamics_world
        sim = org_dynamics_world(n_employees=100, steps=3, seed=1)
        result = sim.run()
        for row in result["org_metrics"]:
            assert row["avg_salary"] > 0

    def test_snapshot_fields_present(self):
        from worldforge.scenarios.org_dynamics import org_dynamics_world
        sim = org_dynamics_world(n_employees=200, steps=12, seed=42)
        result = sim.run()
        snaps = result["employee_snapshot"]
        assert len(snaps) > 0
        assert "role" in snaps[0]
        assert "department" in snaps[0]
        assert "level" in snaps[0]


# ---------------------------------------------------------------------------
# Energy Grid
# ---------------------------------------------------------------------------

class TestEnergyGridWorld:
    def test_runs_and_returns_result(self):
        from worldforge.scenarios.energy_grid import energy_grid_world
        sim = energy_grid_world(n_generators=5, n_consumers=10, n_storage=2, steps=10, seed=42)
        result = sim.run()
        assert "grid_timeseries" in result
        assert "daily_grid_summary" in result

    def test_timeseries_shape(self):
        from worldforge.scenarios.energy_grid import energy_grid_world
        sim = energy_grid_world(n_generators=5, n_consumers=10, n_storage=2, steps=24, seed=42)
        result = sim.run()
        rows = result["grid_timeseries"]
        assert len(rows) == 24   # one row per step
        assert "total_supply_mw" in rows[0]
        assert "total_demand_mw" in rows[0]
        assert "storage_level_mwh" in rows[0]

    def test_supply_nonnegative(self):
        from worldforge.scenarios.energy_grid import energy_grid_world
        sim = energy_grid_world(n_generators=8, n_consumers=20, n_storage=3, steps=12, seed=1)
        result = sim.run()
        for row in result["grid_timeseries"]:
            assert row["total_supply_mw"] >= 0

    def test_storage_charge_events_emitted(self):
        from worldforge.scenarios.energy_grid import energy_grid_world
        sim = energy_grid_world(n_generators=10, n_consumers=20, n_storage=5, steps=48, seed=42)
        result = sim.run()
        events = result["event_log"]
        assert any(e["event_type"] == "StorageChargeEvent" for e in events)

    def test_daily_summary_every_24_steps(self):
        from worldforge.scenarios.energy_grid import energy_grid_world
        sim = energy_grid_world(n_generators=5, n_consumers=10, n_storage=2, steps=48, seed=42)
        result = sim.run()
        summary_rows = result["daily_grid_summary"]
        assert len(summary_rows) == 2   # steps=48, every=24 → 2 summaries
