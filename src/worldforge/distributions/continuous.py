"""Continuous probability distributions."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from worldforge.core.exceptions import DistributionError
from worldforge.distributions.base import Distribution


def _erfinv(z: float) -> float:
    """
    Compute erfinv(z): inverse of the error function.

    Uses scipy when available; falls back to Newton's method otherwise.
    """
    try:
        from scipy.special import erfinv as _scipy_erfinv
        return float(_scipy_erfinv(z))
    except ImportError:
        pass
    # Newton's method: find x such that erf(x) = z
    # Good initial guess using rational approximation
    a = 0.147
    ln = math.log(1 - z * z)
    term = 2 / (math.pi * a) + ln / 2
    x = math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), z)
    for _ in range(10):
        fx = math.erf(x) - z
        dfx = (2 / math.sqrt(math.pi)) * math.exp(-(x * x))
        if dfx == 0:
            break
        x -= fx / dfx
    return x

# Clip type: (low_or_None, high_or_None)
Clip = Optional[tuple[Optional[float], Optional[float]]]


def _apply_clip_scalar(v: float, clip: Clip) -> float:
    if clip is None:
        return v
    lo, hi = clip
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v


def _apply_clip_array(arr: np.ndarray, clip: Clip) -> np.ndarray:
    if clip is None:
        return arr
    lo, hi = clip
    if lo is not None:
        arr = np.maximum(arr, lo)
    if hi is not None:
        arr = np.minimum(arr, hi)
    return arr


class Normal(Distribution):
    """Normal (Gaussian) distribution with optional clipping."""

    def __init__(self, mu: float, sigma: float, clip: Clip = None) -> None:
        if sigma < 0:
            raise DistributionError(f"sigma must be >= 0, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        self.clip = clip

    def sample(self, rng: np.random.Generator) -> float:
        return _apply_clip_scalar(float(rng.normal(self.mu, self.sigma)), self.clip)

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _apply_clip_array(rng.normal(self.mu, self.sigma, size=n), self.clip)

    def mean(self) -> float:
        return self.mu

    def std(self) -> float:
        return self.sigma

    def pdf(self, x: float) -> float:
        if self.sigma == 0:
            return float("inf") if x == self.mu else 0.0
        return math.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (
            self.sigma * math.sqrt(2 * math.pi)
        )

    def cdf(self, x: float) -> float:
        if self.sigma == 0:
            return 1.0 if x >= self.mu else 0.0
        return 0.5 * (1 + math.erf((x - self.mu) / (self.sigma * math.sqrt(2))))

    def ppf(self, q: float) -> float:
        if not 0 < q < 1:
            raise DistributionError(f"q must be in (0, 1), got {q}")
        return self.mu + self.sigma * math.sqrt(2) * _erfinv(2 * q - 1)

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu}, sigma={self.sigma}, clip={self.clip})"


class LogNormal(Distribution):
    """Log-normal distribution. mu and sigma are the underlying normal's parameters."""

    def __init__(self, mu: float, sigma: float, clip: Clip = None) -> None:
        if sigma < 0:
            raise DistributionError(f"sigma must be >= 0, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        self.clip = clip

    def sample(self, rng: np.random.Generator) -> float:
        return _apply_clip_scalar(float(rng.lognormal(self.mu, self.sigma)), self.clip)

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _apply_clip_array(rng.lognormal(self.mu, self.sigma, size=n), self.clip)

    def mean(self) -> float:
        return math.exp(self.mu + 0.5 * self.sigma**2)

    def std(self) -> float:
        variance = (math.exp(self.sigma**2) - 1) * math.exp(2 * self.mu + self.sigma**2)
        return math.sqrt(variance)

    def ppf(self, q: float) -> float:
        if not 0 < q < 1:
            raise DistributionError(f"q must be in (0, 1), got {q}")
        return math.exp(self.mu + self.sigma * math.sqrt(2) * _erfinv(2 * q - 1))

    def __repr__(self) -> str:
        return f"LogNormal(mu={self.mu}, sigma={self.sigma}, clip={self.clip})"


class Exponential(Distribution):
    """Exponential distribution parameterized by scale (mean = scale)."""

    def __init__(self, scale: float, clip: Clip = None) -> None:
        if scale <= 0:
            raise DistributionError(f"scale must be > 0, got {scale}")
        self.scale = scale
        self.clip = clip

    def sample(self, rng: np.random.Generator) -> float:
        return _apply_clip_scalar(float(rng.exponential(self.scale)), self.clip)

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _apply_clip_array(rng.exponential(self.scale, size=n), self.clip)

    def mean(self) -> float:
        return self.scale

    def std(self) -> float:
        return self.scale

    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return math.exp(-x / self.scale) / self.scale

    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return 1 - math.exp(-x / self.scale)

    def ppf(self, q: float) -> float:
        if not 0 < q < 1:
            raise DistributionError(f"q must be in (0, 1), got {q}")
        return -self.scale * math.log(1 - q)

    def __repr__(self) -> str:
        return f"Exponential(scale={self.scale}, clip={self.clip})"


class Pareto(Distribution):
    """Pareto distribution: P(X > x) = (scale/x)^alpha for x >= scale."""

    def __init__(self, alpha: float, scale: float = 1.0) -> None:
        if alpha <= 0:
            raise DistributionError(f"alpha must be > 0, got {alpha}")
        if scale <= 0:
            raise DistributionError(f"scale must be > 0, got {scale}")
        self.alpha = alpha
        self.scale = scale

    def sample(self, rng: np.random.Generator) -> float:
        # numpy pareto gives (X-1) where X is standard Pareto; scale by self.scale
        return float((rng.pareto(self.alpha) + 1) * self.scale)

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return (rng.pareto(self.alpha, size=n) + 1) * self.scale

    def mean(self) -> float:
        if self.alpha <= 1:
            return float("inf")
        return self.alpha * self.scale / (self.alpha - 1)

    def std(self) -> float:
        if self.alpha <= 2:
            return float("inf")
        variance = (self.scale**2 * self.alpha) / ((self.alpha - 1) ** 2 * (self.alpha - 2))
        return math.sqrt(variance)

    def ppf(self, q: float) -> float:
        if not 0 <= q < 1:
            raise DistributionError(f"q must be in [0, 1), got {q}")
        if q == 0:
            return float(self.scale)
        return self.scale / (1 - q) ** (1 / self.alpha)

    def __repr__(self) -> str:
        return f"Pareto(alpha={self.alpha}, scale={self.scale})"


class Gamma(Distribution):
    """Gamma distribution with shape (k) and scale (theta). Mean = shape * scale."""

    def __init__(self, shape: float, scale: float) -> None:
        if shape <= 0:
            raise DistributionError(f"shape must be > 0, got {shape}")
        if scale <= 0:
            raise DistributionError(f"scale must be > 0, got {scale}")
        self.shape = shape
        self.scale = scale

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.gamma(self.shape, self.scale))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(self.shape, self.scale, size=n)

    def mean(self) -> float:
        return self.shape * self.scale

    def std(self) -> float:
        return math.sqrt(self.shape) * self.scale

    def ppf(self, q: float) -> float:
        try:
            from scipy.stats import gamma as scipy_gamma
            return float(scipy_gamma.ppf(q, a=self.shape, scale=self.scale))
        except ImportError:
            raise ImportError("scipy is required for Gamma.ppf(). Install: pip install scipy")

    def __repr__(self) -> str:
        return f"Gamma(shape={self.shape}, scale={self.scale})"


class Beta(Distribution):
    """Beta distribution on [0, 1]. Mean = alpha / (alpha + beta)."""

    def __init__(self, alpha: float, beta: float) -> None:
        if alpha <= 0:
            raise DistributionError(f"alpha must be > 0, got {alpha}")
        if beta <= 0:
            raise DistributionError(f"beta must be > 0, got {beta}")
        self.alpha = alpha
        self.beta = beta

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size=n)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def std(self) -> float:
        ab = self.alpha + self.beta
        return math.sqrt(self.alpha * self.beta / (ab**2 * (ab + 1)))

    def ppf(self, q: float) -> float:
        try:
            from scipy.stats import beta as scipy_beta
            return float(scipy_beta.ppf(q, self.alpha, self.beta))
        except ImportError:
            raise ImportError("scipy is required for Beta.ppf(). Install: pip install scipy")

    def __repr__(self) -> str:
        return f"Beta(alpha={self.alpha}, beta={self.beta})"


class Uniform(Distribution):
    """Uniform distribution on [low, high]."""

    def __init__(self, low: float, high: float) -> None:
        if low >= high:
            raise DistributionError(f"low must be < high, got low={low}, high={high}")
        self.low = low
        self.high = high

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=n)

    def mean(self) -> float:
        return (self.low + self.high) / 2

    def std(self) -> float:
        return (self.high - self.low) / math.sqrt(12)

    def pdf(self, x: float) -> float:
        if self.low <= x <= self.high:
            return 1.0 / (self.high - self.low)
        return 0.0

    def cdf(self, x: float) -> float:
        if x < self.low:
            return 0.0
        if x > self.high:
            return 1.0
        return (x - self.low) / (self.high - self.low)

    def ppf(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise DistributionError(f"q must be in [0, 1], got {q}")
        return self.low + (self.high - self.low) * q

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"


class Triangular(Distribution):
    """Triangular distribution with lower, mode, and upper bounds."""

    def __init__(self, low: float, mode: float, high: float) -> None:
        if not (low <= mode <= high):
            raise DistributionError(
                f"Must satisfy low <= mode <= high, got low={low}, mode={mode}, high={high}"
            )
        if low == high:
            raise DistributionError("low and high must be different")
        self.low = low
        self.mode = mode
        self.high = high

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.triangular(self.low, self.mode, self.high))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, size=n)

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3

    def std(self) -> float:
        lo, c, hi = self.low, self.mode, self.high
        variance = (lo**2 + c**2 + hi**2 - lo * c - lo * hi - c * hi) / 18
        return math.sqrt(variance)

    def __repr__(self) -> str:
        return f"Triangular(low={self.low}, mode={self.mode}, high={self.high})"


class Weibull(Distribution):
    """Weibull distribution with shape (k) and scale (lambda)."""

    def __init__(self, shape: float, scale: float = 1.0) -> None:
        if shape <= 0:
            raise DistributionError(f"shape must be > 0, got {shape}")
        if scale <= 0:
            raise DistributionError(f"scale must be > 0, got {scale}")
        self.shape = shape
        self.scale = scale

    def sample(self, rng: np.random.Generator) -> float:
        # numpy weibull samples from standard Weibull(shape); multiply by scale
        return float(rng.weibull(self.shape) * self.scale)

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.weibull(self.shape, size=n) * self.scale

    def mean(self) -> float:
        return self.scale * math.gamma(1 + 1 / self.shape)

    def std(self) -> float:
        variance = self.scale**2 * (
            math.gamma(1 + 2 / self.shape) - math.gamma(1 + 1 / self.shape) ** 2
        )
        return math.sqrt(variance)

    def ppf(self, q: float) -> float:
        if not 0 < q < 1:
            raise DistributionError(f"q must be in (0, 1), got {q}")
        return self.scale * (-math.log(1 - q)) ** (1 / self.shape)

    def __repr__(self) -> str:
        return f"Weibull(shape={self.shape}, scale={self.scale})"
