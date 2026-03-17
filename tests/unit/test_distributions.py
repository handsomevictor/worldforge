"""Tests for all distribution implementations (Steps 03-06)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from worldforge.distributions import (
    Normal, LogNormal, Exponential, Pareto, Gamma, Beta,
    Uniform, Triangular, Weibull,
    Poisson, Binomial, Geometric, Empirical, Categorical,
    HourOfDay, DayOfWeek, Seasonal,
    MixtureDistribution, ConditionalDistribution, CorrelatedDistributions,
)
from worldforge.core.exceptions import DistributionError

N_SAMPLES = 100_000
TOLERANCE = 0.02  # 2% for moments (generous for skewed distributions)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_moments(dist, rng, *, rtol=TOLERANCE, n=N_SAMPLES):
    """Assert sampled mean and std are within rtol of analytical values."""
    samples = dist.sample_batch(n, rng)
    expected_mean = dist.mean()
    expected_std = dist.std()

    if not math.isinf(expected_mean):
        assert abs(np.mean(samples) - expected_mean) / max(abs(expected_mean), 1e-9) < rtol, (
            f"{dist}: mean={np.mean(samples):.4f}, expected={expected_mean:.4f}"
        )
    if not math.isinf(expected_std) and expected_std > 0:
        assert abs(np.std(samples) - expected_std) / max(expected_std, 1e-9) < rtol, (
            f"{dist}: std={np.std(samples):.4f}, expected={expected_std:.4f}"
        )


# ---------------------------------------------------------------------------
# Continuous distributions
# ---------------------------------------------------------------------------

class TestNormal:
    def test_moments(self, rng):
        check_moments(Normal(mu=10, sigma=2), rng)

    def test_clip_lower(self, rng):
        d = Normal(mu=0, sigma=5, clip=(0, None))
        samples = d.sample_batch(10_000, rng)
        assert samples.min() >= 0

    def test_clip_both(self, rng):
        d = Normal(mu=0, sigma=10, clip=(-1, 1))
        samples = d.sample_batch(10_000, rng)
        assert samples.min() >= -1
        assert samples.max() <= 1

    def test_pdf_peak_at_mean(self):
        d = Normal(mu=5, sigma=1)
        # PDF is highest at the mean
        assert d.pdf(5) > d.pdf(3)
        assert d.pdf(5) > d.pdf(7)

    def test_cdf_at_mean(self):
        d = Normal(mu=0, sigma=1)
        assert abs(d.cdf(0) - 0.5) < 1e-9

    def test_ppf_roundtrip(self):
        d = Normal(mu=3, sigma=2)
        for q in [0.1, 0.5, 0.9]:
            x = d.ppf(q)
            assert abs(d.cdf(x) - q) < 1e-6

    def test_invalid_sigma(self):
        with pytest.raises(DistributionError):
            Normal(mu=0, sigma=-1)

    def test_ppf_invalid_q(self):
        d = Normal(mu=0, sigma=1)
        with pytest.raises(DistributionError):
            d.ppf(0)
        with pytest.raises(DistributionError):
            d.ppf(1)


class TestLogNormal:
    def test_moments(self, rng):
        check_moments(LogNormal(mu=1, sigma=0.5), rng)

    def test_positive_samples(self, rng):
        samples = LogNormal(mu=0, sigma=1).sample_batch(10_000, rng)
        assert np.all(samples > 0)

    def test_ppf_roundtrip(self):
        d = LogNormal(mu=2, sigma=0.3)
        for q in [0.1, 0.5, 0.9]:
            x = d.ppf(q)
            # Verify: ln(x) should satisfy normal ppf
            expected = Normal(d.mu, d.sigma).ppf(q)
            assert abs(math.log(x) - expected) < 1e-6


class TestExponential:
    def test_moments(self, rng):
        check_moments(Exponential(scale=5), rng)

    def test_non_negative(self, rng):
        samples = Exponential(scale=1).sample_batch(10_000, rng)
        assert np.all(samples >= 0)

    def test_ppf_roundtrip(self):
        d = Exponential(scale=3)
        for q in [0.1, 0.5, 0.9]:
            x = d.ppf(q)
            assert abs(d.cdf(x) - q) < 1e-9

    def test_invalid_scale(self):
        with pytest.raises(DistributionError):
            Exponential(scale=0)
        with pytest.raises(DistributionError):
            Exponential(scale=-1)


class TestPareto:
    def test_moments(self, rng):
        # alpha > 2 for finite variance
        check_moments(Pareto(alpha=4, scale=1), rng, rtol=0.05)

    def test_minimum_is_scale(self, rng):
        d = Pareto(alpha=2, scale=5)
        samples = d.sample_batch(10_000, rng)
        assert np.all(samples >= 5)

    def test_ppf(self):
        d = Pareto(alpha=2, scale=1)
        assert abs(d.ppf(0.0) - 1.0) < 1e-9  # minimum value at q=0
        assert d.ppf(0.99) > 1.0


class TestUniform:
    def test_moments(self, rng):
        check_moments(Uniform(low=2, high=8), rng)

    def test_bounds(self, rng):
        samples = Uniform(low=3, high=7).sample_batch(10_000, rng)
        assert samples.min() >= 3
        assert samples.max() <= 7

    def test_ppf_roundtrip(self):
        d = Uniform(low=1, high=5)
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert abs(d.cdf(d.ppf(q)) - q) < 1e-9

    def test_invalid_bounds(self):
        with pytest.raises(DistributionError):
            Uniform(low=5, high=5)
        with pytest.raises(DistributionError):
            Uniform(low=5, high=3)


class TestTriangular:
    def test_moments(self, rng):
        check_moments(Triangular(low=1, mode=3, high=6), rng)

    def test_bounds(self, rng):
        d = Triangular(low=0, mode=2, high=4)
        samples = d.sample_batch(10_000, rng)
        assert samples.min() >= 0
        assert samples.max() <= 4


class TestWeibull:
    def test_moments(self, rng):
        check_moments(Weibull(shape=2, scale=3), rng, rtol=0.03)

    def test_positive(self, rng):
        samples = Weibull(shape=1.5, scale=2).sample_batch(10_000, rng)
        assert np.all(samples > 0)

    def test_ppf_roundtrip(self):
        d = Weibull(shape=2, scale=3)
        for q in [0.1, 0.5, 0.9]:
            x = d.ppf(q)
            # Verify CDF(PPF(q)) ≈ q via direct formula
            computed_q = 1 - math.exp(-(x / d.scale) ** d.shape)
            assert abs(computed_q - q) < 1e-9


class TestGamma:
    def test_moments(self, rng):
        check_moments(Gamma(shape=3, scale=2), rng)

    def test_positive(self, rng):
        samples = Gamma(shape=2, scale=1).sample_batch(10_000, rng)
        assert np.all(samples > 0)


class TestBeta:
    def test_moments(self, rng):
        check_moments(Beta(alpha=2, beta=5), rng)

    def test_unit_interval(self, rng):
        samples = Beta(alpha=1, beta=1).sample_batch(10_000, rng)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)


# ---------------------------------------------------------------------------
# Discrete distributions
# ---------------------------------------------------------------------------

class TestPoisson:
    def test_moments(self, rng):
        check_moments(Poisson(lam=10), rng)

    def test_non_negative(self, rng):
        samples = Poisson(lam=5).sample_batch(10_000, rng)
        assert np.all(samples >= 0)

    def test_integer_samples(self, rng):
        samples = Poisson(lam=3).sample_batch(1000, rng)
        assert np.all(samples == samples.astype(int))

    def test_invalid_lam(self):
        with pytest.raises(DistributionError):
            Poisson(lam=-1)


class TestBinomial:
    def test_moments(self, rng):
        check_moments(Binomial(n=100, p=0.3), rng)

    def test_bounds(self, rng):
        d = Binomial(n=10, p=0.5)
        samples = d.sample_batch(10_000, rng)
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)


class TestGeometric:
    def test_moments(self, rng):
        check_moments(Geometric(p=0.3), rng, rtol=0.05)

    def test_positive(self, rng):
        samples = Geometric(p=0.5).sample_batch(10_000, rng)
        assert np.all(samples >= 1)

    def test_ppf(self):
        d = Geometric(p=0.5)
        # P(X <= 1) = 0.5, so ppf(0.5) = 1
        assert d.ppf(0.5) == 1


class TestEmpirical:
    def test_samples_from_values(self, rng):
        d = Empirical(values=[1, 2, 3])
        samples = d.sample_batch(1000, rng)
        assert set(samples).issubset({1, 2, 3})

    def test_weighted(self, rng):
        # value 10 should appear ~90% of the time
        d = Empirical(values=[10, 20], weights=[9, 1])
        samples = d.sample_batch(10_000, rng)
        rate_10 = np.mean(samples == 10)
        assert abs(rate_10 - 0.9) < 0.02

    def test_from_data(self, rng):
        data = [5, 5, 5, 10, 10]
        d = Empirical.from_data(data)
        samples = d.sample_batch(10_000, rng)
        assert set(samples).issubset({5, 10})

    def test_mean(self):
        d = Empirical(values=[1, 2, 3], weights=[1, 1, 1])
        assert abs(d.mean() - 2.0) < 1e-9

    def test_empty_values(self):
        with pytest.raises(DistributionError):
            Empirical(values=[])


class TestCategorical:
    def test_samples_from_choices(self, rng):
        d = Categorical(choices=["a", "b", "c"], weights=[1, 1, 1])
        samples = d.sample_batch(1000, rng)
        assert set(samples).issubset({"a", "b", "c"})

    def test_weighted(self, rng):
        # "x" should appear ~80% of the time
        d = Categorical(choices=["x", "y"], weights=[8, 2])
        samples = d.sample_batch(10_000, rng)
        rate_x = np.mean(samples == "x")
        assert abs(rate_x - 0.8) < 0.02

    def test_empty_choices(self):
        with pytest.raises(DistributionError):
            Categorical(choices=[])


# ---------------------------------------------------------------------------
# Temporal distributions
# ---------------------------------------------------------------------------

class TestHourOfDay:
    def test_multiplier_at_defined_hour(self):
        h = HourOfDay(pattern={8: 0.5, 12: 1.2, 18: 1.5})
        # Simulate a datetime-like object
        class FakeNow:
            hour = 12
        assert h.get_multiplier(FakeNow()) == 1.2

    def test_multiplier_between_hours(self):
        h = HourOfDay(pattern={8: 0.5, 12: 1.2})
        class FakeNow:
            hour = 10  # between 8 and 12, should use 8's value
        assert h.get_multiplier(FakeNow()) == 0.5

    def test_multiplier_none_returns_one(self):
        h = HourOfDay(pattern={8: 2.0})
        assert h.get_multiplier(None) == 1.0


class TestDayOfWeek:
    def test_by_name(self):
        d = DayOfWeek(pattern={"Mon": 1.0, "Sat": 1.5, "Sun": 0.8})
        class FakeNow:
            def weekday(self): return 5  # Saturday
        assert d.get_multiplier(FakeNow()) == 1.5

    def test_missing_day_defaults_to_one(self):
        d = DayOfWeek(pattern={"Mon": 2.0})
        class FakeNow:
            def weekday(self): return 3  # Thursday, not defined
        assert d.get_multiplier(FakeNow()) == 1.0

    def test_none_returns_one(self):
        d = DayOfWeek(pattern={"Mon": 2.0})
        assert d.get_multiplier(None) == 1.0


class TestSeasonal:
    def test_passthrough_without_context(self, rng):
        # Without now, multiplier = 1.0, so mean should match base
        base = Poisson(lam=100)
        d = Seasonal(base=base, hour_multiplier=HourOfDay(pattern={0: 2.0}))
        samples = d.sample_batch(10_000, rng, now=None)
        assert abs(np.mean(samples) - 100) / 100 < 0.05

    def test_multiplier_applied(self, rng):
        base = Poisson(lam=100)
        hour_mult = HourOfDay(pattern={12: 2.0})

        class FakeNow:
            hour = 12
            def weekday(self): return 1
        d = Seasonal(base=base, hour_multiplier=hour_mult)
        samples = d.sample_batch(10_000, rng, now=FakeNow())
        # Mean should be ~200 (100 * 2.0)
        assert abs(np.mean(samples) - 200) / 200 < 0.05


# ---------------------------------------------------------------------------
# MixtureDistribution
# ---------------------------------------------------------------------------

class TestMixtureDistribution:
    def test_samples_from_components(self, rng):
        # Two very separated components
        d = MixtureDistribution(
            components=[Normal(mu=-100, sigma=1), Normal(mu=100, sigma=1)],
            weights=[0.5, 0.5],
        )
        samples = d.sample_batch(1000, rng)
        # Every sample should be either near -100 or near 100
        assert np.all((samples < -50) | (samples > 50))

    def test_weighted_mean(self, rng):
        # 80% from Normal(0,1), 20% from Normal(100,1) → mean ≈ 20
        d = MixtureDistribution(
            components=[Normal(0, 1), Normal(100, 1)],
            weights=[0.8, 0.2],
        )
        check_moments_mean_only(d, rng, expected_mean=20.0)

    def test_invalid_empty(self):
        with pytest.raises(DistributionError):
            MixtureDistribution(components=[], weights=[])

    def test_invalid_weight_mismatch(self):
        with pytest.raises(DistributionError):
            MixtureDistribution(components=[Normal(0, 1)], weights=[0.5, 0.5])


def check_moments_mean_only(dist, rng, expected_mean, rtol=0.03, n=N_SAMPLES):
    samples = dist.sample_batch(n, rng)
    assert abs(np.mean(samples) - expected_mean) / max(abs(expected_mean), 1) < rtol


# ---------------------------------------------------------------------------
# ConditionalDistribution
# ---------------------------------------------------------------------------

class TestConditionalDistribution:
    def test_correct_component_selected(self, rng):
        d = ConditionalDistribution(
            condition=lambda ctx: ctx,
            mapping={
                "low":  Uniform(0, 1),
                "high": Uniform(100, 101),
            },
        )
        low_samples = d.sample_batch(1000, rng, context="low")
        high_samples = d.sample_batch(1000, rng, context="high")
        assert np.all(low_samples <= 1)
        assert np.all(high_samples >= 100)

    def test_missing_key_raises(self, rng):
        d = ConditionalDistribution(
            condition=lambda ctx: ctx,
            mapping={"a": Normal(0, 1)},
        )
        with pytest.raises(DistributionError):
            d.sample(rng, context="b")

    def test_default_used_when_key_missing(self, rng):
        fallback = Uniform(999, 1000)
        d = ConditionalDistribution(
            condition=lambda ctx: ctx,
            mapping={"a": Normal(0, 1)},
            default=fallback,
        )
        s = d.sample(rng, context="unknown")
        assert 999 <= s <= 1000

    def test_no_context_returns_fallback(self, rng):
        d = ConditionalDistribution(
            condition=lambda ctx: ctx,
            mapping={"a": Uniform(0, 1), "b": Uniform(10, 11)},
        )
        # No context: returns first mapping value without error
        d.sample(rng, context=None)


# ---------------------------------------------------------------------------
# CorrelatedDistributions
# ---------------------------------------------------------------------------

class TestCorrelatedDistributions:
    def test_output_shape(self, rng):
        d = CorrelatedDistributions(
            distributions=[Normal(0, 1), Normal(0, 1)],
            correlation=0.8,
        )
        result = d.sample(rng)
        assert len(result) == 2

    def test_batch_output_shape(self, rng):
        d = CorrelatedDistributions(
            distributions=[Normal(0, 1), Normal(0, 1), Normal(0, 1)],
            correlation=0.5,
        )
        result = d.sample_batch(100, rng)
        assert result.shape == (100, 3)

    def test_negative_correlation(self, rng):
        # Strongly negative correlation: when X is large, Y should be small
        d = CorrelatedDistributions(
            distributions=[Normal(0, 1), Normal(0, 1)],
            correlation=-0.9,
        )
        samples = d.sample_batch(5000, rng)
        corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert corr < -0.7, f"Expected correlation < -0.7, got {corr:.3f}"

    def test_positive_correlation(self, rng):
        d = CorrelatedDistributions(
            distributions=[Normal(0, 1), Normal(0, 1)],
            correlation=0.9,
        )
        samples = d.sample_batch(5000, rng)
        corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert corr > 0.7, f"Expected correlation > 0.7, got {corr:.3f}"

    def test_invalid_matrix(self):
        # Not positive definite
        with pytest.raises(DistributionError):
            CorrelatedDistributions(
                distributions=[Normal(0, 1), Normal(0, 1)],
                correlation=2.0,  # invalid: must be in (-1, 1)
            )

    def test_too_few_distributions(self):
        with pytest.raises(DistributionError):
            CorrelatedDistributions(distributions=[Normal(0, 1)], correlation=0.5)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_samples(self):
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        d = Normal(mu=5, sigma=2)
        samples_a = d.sample_batch(1000, rng_a)
        samples_b = d.sample_batch(1000, rng_b)
        np.testing.assert_array_equal(samples_a, samples_b)

    def test_different_seed_different_samples(self):
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        d = Normal(mu=0, sigma=1)
        samples_a = d.sample_batch(100, rng_a)
        samples_b = d.sample_batch(100, rng_b)
        assert not np.array_equal(samples_a, samples_b)
