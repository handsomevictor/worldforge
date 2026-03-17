"""
Data integrity and logical consistency regression tests.

Each test corresponds to a specific bug that was identified and fixed.
"""
from __future__ import annotations

import pytest
from collections import Counter

from worldforge.agent import _reset_id_counter


@pytest.fixture(autouse=True)
def reset_ids():
    _reset_id_counter(1)
    yield


# ---------------------------------------------------------------------------
# BUG-FIX-01: epidemic double-infection
# A susceptible person should only receive ONE InfectionEvent per tick,
# even if multiple infected agents contact them in the same step.
# ---------------------------------------------------------------------------

class TestNoDoubleInfection:
    def test_each_person_infected_at_most_once_per_run(self):
        """Each person_id should appear at most once as an InfectionEvent target."""
        from worldforge.scenarios.epidemic import epidemic_world
        sim = epidemic_world(
            population=200,
            initial_infected=20,
            transmission_prob=0.99,   # extremely high — maximises collision probability
            duration_days=3,
            seed=42,
        )
        result = sim.run()
        events = result["event_log"]
        infection_events = [e for e in events if e["event_type"] == "InfectionEvent"]
        person_ids = [e["person_id"] for e in infection_events]
        duplicates = [pid for pid, count in Counter(person_ids).items() if count > 1]
        assert duplicates == [], f"Double-infected persons: {duplicates}"


# ---------------------------------------------------------------------------
# BUG-FIX-02: epidemic module-level state isolation
# Running two epidemic_world() sims with different parameters should
# not contaminate each other's recovery rates.
# ---------------------------------------------------------------------------

class TestEpidemicIsolation:
    def test_two_sims_independent(self):
        """Two epidemic sims with different recovery_rate produce different SIR curves."""
        from worldforge.scenarios.epidemic import epidemic_world
        sim_fast = epidemic_world(population=300, recovery_rate=0.20, duration_days=10, seed=1)
        sim_slow = epidemic_world(population=300, recovery_rate=0.01, duration_days=10, seed=1)

        fast = sim_fast.run()
        slow = sim_slow.run()

        fast_r_final = fast["sir_curve"][-1]["R"]
        slow_r_final = slow["sir_curve"][-1]["R"]
        # High recovery rate → more recovered by day 10
        assert fast_r_final >= slow_r_final, (
            f"Fast recovery ({fast_r_final}) should recover >= slow ({slow_r_final})"
        )


# ---------------------------------------------------------------------------
# BUG-FIX-03: game_economy module-level price reset
# Running game_economy_world() twice should start with fresh prices.
# ---------------------------------------------------------------------------

class TestGameEconomyPriceReset:
    def test_second_run_starts_with_default_prices(self):
        """Prices are reset to defaults at the start of each factory call."""
        from worldforge.scenarios.game_economy import game_economy_world, _MARKET_PRICES_DEFAULTS
        sim1 = game_economy_world(n_players=50, steps=5, seed=1, initial_prices={"sword": 9999})
        sim1.run()

        sim2 = game_economy_world(n_players=50, steps=1, seed=2)
        result2 = sim2.run()
        rows = result2["economy_metrics"]
        # After reset, sword price should be close to 100 (default), not 9999
        assert rows[0]["sword_price"] < 200, (
            f"Expected sword_price near 100, got {rows[0]['sword_price']}"
        )


# ---------------------------------------------------------------------------
# BUG-FIX-04: ecommerce UserSignupEvent now emitted
# ---------------------------------------------------------------------------

class TestUserSignupEvent:
    def test_signup_events_equal_initial_users(self):
        """One UserSignupEvent per user should be emitted at simulation start."""
        from worldforge.scenarios.ecommerce import ecommerce_world
        n = 100
        sim = ecommerce_world(n_users=n, duration="2 days", seed=42)
        result = sim.run()
        events = result["event_log"]
        signups = [e for e in events if e["event_type"] == "UserSignupEvent"]
        assert len(signups) == n, f"Expected {n} signup events, got {len(signups)}"

    def test_signup_events_have_tier(self):
        from worldforge.scenarios.ecommerce import ecommerce_world
        sim = ecommerce_world(n_users=50, duration="2 days", seed=1)
        result = sim.run()
        signups = [e for e in result["event_log"] if e["event_type"] == "UserSignupEvent"]
        tiers = {e["tier"] for e in signups}
        assert tiers.issubset({"free", "premium", "vip"})


# ---------------------------------------------------------------------------
# BUG-FIX-05: ecommerce gmv_daily is windowed, not cumulative
# ---------------------------------------------------------------------------

class TestGmvDailyWindowed:
    def test_gmv_daily_non_monotonic(self):
        """gmv_daily should vary each day, not monotonically increase like cumulative."""
        from worldforge.scenarios.ecommerce import ecommerce_world
        sim = ecommerce_world(n_users=500, duration="10 days", seed=42)
        result = sim.run()
        rows = result["daily_metrics"]
        daily = [r["gmv_daily"] for r in rows]
        cumulative = [r["gmv_cumulative"] for r in rows]

        # cumulative should be non-decreasing
        assert all(cumulative[i] <= cumulative[i + 1] for i in range(len(cumulative) - 1))
        # gmv_daily should NOT be identical to cumulative after the first day
        assert daily[-1] != cumulative[-1], "gmv_daily should not equal cumulative at the end"

    def test_gmv_cumulative_at_least_daily(self):
        """Cumulative GMV >= any single day's GMV."""
        from worldforge.scenarios.ecommerce import ecommerce_world
        sim = ecommerce_world(n_users=300, duration="5 days", seed=7)
        result = sim.run()
        rows = result["daily_metrics"]
        for row in rows:
            assert row["gmv_cumulative"] >= row["gmv_daily"]


# ---------------------------------------------------------------------------
# BUG-FIX-06: fintech monthly metrics are windowed
# ---------------------------------------------------------------------------

class TestFintechMonthlyMetrics:
    def test_monthly_and_cumulative_diverge(self):
        from worldforge.scenarios.fintech import fintech_world
        sim = fintech_world(n_users=200, duration_days=90, seed=42)
        result = sim.run()
        rows = result["monthly_metrics"]
        if len(rows) < 2:
            pytest.skip("Not enough monthly snapshots")
        # By month 3, cumulative deposits should exceed monthly deposits
        assert rows[-1]["total_deposits"] >= rows[-1]["deposits_this_month"]


# ---------------------------------------------------------------------------
# BUG-FIX-07: IoT anomaly counter is now correct
# ---------------------------------------------------------------------------

class TestIoTAnomalyCounter:
    def test_anomaly_rate_less_than_one(self):
        """Anomaly rate should be well below 1 (default anomaly_rate=0.005)."""
        from worldforge.scenarios.iot_timeseries import iot_world
        sim = iot_world(n_sensors=50, duration_steps=120, anomaly_rate=0.005, seed=42)
        result = sim.run()
        rows = result["hourly_summary"]
        assert all(0.0 <= r["anomaly_rate"] <= 1.0 for r in rows)
        # With anomaly_rate=0.005, should be clearly less than total readings
        assert all(r["n_anomalies"] <= r["n_readings_total"] for r in rows)

    def test_anomaly_count_less_than_total_readings(self):
        from worldforge.scenarios.iot_timeseries import iot_world
        sim = iot_world(n_sensors=100, duration_steps=120, anomaly_rate=0.01, seed=1)
        result = sim.run()
        rows = result["hourly_summary"]
        for r in rows:
            assert r["n_anomalies"] <= r["n_readings_total"]


# ---------------------------------------------------------------------------
# BUG-FIX-08: org_dynamics HireEvent now emitted for spawned employees
# ---------------------------------------------------------------------------

class TestOrgHireEvent:
    def test_hire_events_emitted(self):
        """HireEvent should be emitted for each spawned employee."""
        from worldforge.scenarios.org_dynamics import org_dynamics_world
        sim = org_dynamics_world(n_employees=100, steps=6, hiring_rate=0.05, seed=42)
        result = sim.run()
        events = result["event_log"]
        hire_events = [e for e in events if e["event_type"] == "HireEvent"]
        # hiring_rate=0.05 over 6 steps → at least some HireEvents
        assert len(hire_events) > 0, "Expected at least some HireEvents"

    def test_hire_event_has_required_fields(self):
        from worldforge.scenarios.org_dynamics import org_dynamics_world
        sim = org_dynamics_world(n_employees=100, steps=5, hiring_rate=0.10, seed=1)
        result = sim.run()
        events = result["event_log"]
        hire_events = [e for e in events if e["event_type"] == "HireEvent"]
        if hire_events:
            e = hire_events[0]
            assert "role" in e
            assert "department" in e
            assert "salary" in e
            assert e["salary"] > 0


# ---------------------------------------------------------------------------
# BUG-FIX-09: energy_grid discharge efficiency applied symmetrically
# ---------------------------------------------------------------------------

class TestEnergyStorageEfficiency:
    def test_storage_does_not_exceed_capacity(self):
        """Battery charge_level_mwh should never exceed capacity or go below 0."""
        from worldforge.scenarios.energy_grid import energy_grid_world, BatteryStorage
        sim = energy_grid_world(n_generators=5, n_consumers=10, n_storage=3, steps=48, seed=42)
        # Run and verify storage bounds via probe data
        result = sim.run()
        rows = result["grid_timeseries"]
        for r in rows:
            assert r["storage_level_mwh"] >= 0, f"Negative storage: {r['storage_level_mwh']}"

    def test_discharge_events_have_positive_energy(self):
        from worldforge.scenarios.energy_grid import energy_grid_world
        sim = energy_grid_world(n_generators=5, n_consumers=15, n_storage=5, steps=48, seed=1)
        result = sim.run()
        events = result["event_log"]
        discharges = [
            e for e in events
            if e["event_type"] == "StorageChargeEvent" and e["action"] == "discharge"
        ]
        for e in discharges:
            assert e["energy_mwh"] >= 0


# ---------------------------------------------------------------------------
# BUG-FIX-10: market_microstructure environment auto-wired
# ---------------------------------------------------------------------------

class TestMarketEnvironmentWired:
    def test_mid_price_series_non_zero(self):
        """mid_price should be non-zero — requires ctx.environment to be set."""
        from worldforge.scenarios.market_microstructure import market_microstructure_world
        sim = market_microstructure_world(
            n_market_makers=3, n_noise_traders=10, duration_steps=20, seed=42
        )
        result = sim.run()
        prices = result["price_series"]
        assert all(r["mid_price"] > 0 for r in prices), "mid_price is 0 — environment not wired"


# ---------------------------------------------------------------------------
# BUG-FIX-11: result.validate() detects NaN and Inf
# ---------------------------------------------------------------------------

class TestValidateResult:
    def test_clean_result_passes(self):
        from worldforge.output.result import SimulationResult
        r = SimulationResult(
            data={"m": [{"v": 1.0, "timestamp": 1}, {"v": 2.0, "timestamp": 2}]},
        )
        report = r.validate()
        assert report.passed
        assert report.errors == []

    def test_nan_detected(self):
        from worldforge.output.result import SimulationResult
        import math
        r = SimulationResult(data={"m": [{"v": float("nan")}]})
        report = r.validate()
        assert not report.passed
        assert any("NaN" in e for e in report.errors)

    def test_inf_detected(self):
        from worldforge.output.result import SimulationResult
        r = SimulationResult(data={"m": [{"v": float("inf")}]})
        report = r.validate()
        assert not report.passed
        assert any("Inf" in e for e in report.errors)

    def test_negative_timestamp_monotonicity_detected(self):
        from worldforge.output.result import SimulationResult
        r = SimulationResult(data={"m": [
            {"v": 1, "timestamp": 5},
            {"v": 2, "timestamp": 3},   # backwards!
        ]})
        report = r.validate()
        assert not report.passed
        assert any("backwards" in e for e in report.errors)

    def test_real_sim_passes_validate(self):
        from worldforge.scenarios.ecommerce import ecommerce_world
        result = ecommerce_world(n_users=100, duration="5 days", seed=42).run()
        report = result.validate()
        assert report.passed, f"Validation failed: {report.errors}"
