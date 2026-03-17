"""Discrete probability distributions."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from worldforge.core.exceptions import DistributionError
from worldforge.distributions.base import Distribution


class Poisson(Distribution):
    """Poisson distribution with rate parameter lam."""

    def __init__(self, lam: float) -> None:
        if lam < 0:
            raise DistributionError(f"lam must be >= 0, got {lam}")
        self.lam = lam

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.poisson(self.lam))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(self.lam, size=n)

    def mean(self) -> float:
        return self.lam

    def std(self) -> float:
        return math.sqrt(self.lam)

    def ppf(self, q: float) -> int:
        try:
            from scipy.stats import poisson as scipy_poisson
            return int(scipy_poisson.ppf(q, self.lam))
        except ImportError:
            raise ImportError("scipy is required for Poisson.ppf(). Install: pip install scipy")

    def __repr__(self) -> str:
        return f"Poisson(lam={self.lam})"


class Binomial(Distribution):
    """Binomial distribution: number of successes in n trials with probability p."""

    def __init__(self, n: int, p: float) -> None:
        if n < 1:
            raise DistributionError(f"n must be >= 1, got {n}")
        if not 0 <= p <= 1:
            raise DistributionError(f"p must be in [0, 1], got {p}")
        self.n = n
        self.p = p

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.binomial(self.n, self.p))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.binomial(self.n, self.p, size=n)

    def mean(self) -> float:
        return self.n * self.p

    def std(self) -> float:
        return math.sqrt(self.n * self.p * (1 - self.p))

    def __repr__(self) -> str:
        return f"Binomial(n={self.n}, p={self.p})"


class Geometric(Distribution):
    """Geometric distribution: number of trials until first success (inclusive)."""

    def __init__(self, p: float) -> None:
        if not 0 < p <= 1:
            raise DistributionError(f"p must be in (0, 1], got {p}")
        self.p = p

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.geometric(self.p))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.geometric(self.p, size=n)

    def mean(self) -> float:
        return 1 / self.p

    def std(self) -> float:
        return math.sqrt((1 - self.p) / self.p**2)

    def ppf(self, q: float) -> int:
        if not 0 < q < 1:
            raise DistributionError(f"q must be in (0, 1), got {q}")
        # ceil(log(1-q) / log(1-p))
        return math.ceil(math.log(1 - q) / math.log(1 - self.p))

    def __repr__(self) -> str:
        return f"Geometric(p={self.p})"


class Empirical(Distribution):
    """
    Empirical distribution sampled from explicit values with optional weights.

    Use Empirical.from_data(data) to build from a raw dataset.
    """

    def __init__(
        self,
        values: Sequence,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if len(values) == 0:
            raise DistributionError("values must not be empty")
        self._values = list(values)
        if weights is not None:
            if len(weights) != len(values):
                raise DistributionError("weights and values must have the same length")
            total = sum(weights)
            self._probs = [w / total for w in weights]
        else:
            self._probs = None

    @classmethod
    def from_data(cls, data: Sequence) -> "Empirical":
        """Build an unweighted empirical distribution from a dataset."""
        return cls(values=data)

    def sample(self, rng: np.random.Generator):
        idx = rng.choice(len(self._values), p=self._probs)
        return self._values[idx]

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(len(self._values), size=n, p=self._probs)
        return np.array([self._values[i] for i in indices])

    def mean(self) -> float:
        values = np.array(self._values, dtype=float)
        if self._probs is not None:
            return float(np.dot(values, self._probs))
        return float(np.mean(values))

    def std(self) -> float:
        values = np.array(self._values, dtype=float)
        if self._probs is not None:
            m = self.mean()
            variance = float(np.dot(self._probs, (values - m) ** 2))
            return math.sqrt(variance)
        return float(np.std(values))

    def __repr__(self) -> str:
        return f"Empirical(n_values={len(self._values)})"


class Categorical(Distribution):
    """
    Categorical distribution over a discrete set of choices.

    Samples return one of the choices with the given probabilities.
    """

    def __init__(
        self,
        choices: Sequence,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if len(choices) == 0:
            raise DistributionError("choices must not be empty")
        self._choices = list(choices)
        if weights is not None:
            if len(weights) != len(choices):
                raise DistributionError("weights and choices must have the same length")
            total = sum(weights)
            self._probs = [w / total for w in weights]
        else:
            self._probs = None

    def sample(self, rng: np.random.Generator):
        idx = int(rng.choice(len(self._choices), p=self._probs))
        return self._choices[idx]

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(len(self._choices), size=n, p=self._probs)
        return np.array([self._choices[i] for i in indices])

    def __repr__(self) -> str:
        return f"Categorical(choices={self._choices}, probs={self._probs})"
