"""Abstract base class for all probability distributions."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):
    """
    Abstract base for all worldforge probability distributions.

    All subclasses must implement sample(). The batch version sample_batch()
    has a default implementation that calls sample() in a loop, but subclasses
    should override it with a vectorized numpy implementation for performance.
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float:
        """Draw a single sample using the provided RNG."""

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples. Override with vectorized implementation when possible."""
        return np.array([self.sample(rng) for _ in range(n)])

    def mean(self) -> float:
        raise NotImplementedError(f"{type(self).__name__} does not implement mean()")

    def std(self) -> float:
        raise NotImplementedError(f"{type(self).__name__} does not implement std()")

    def pdf(self, x: float) -> float:
        raise NotImplementedError(f"{type(self).__name__} does not implement pdf()")

    def cdf(self, x: float) -> float:
        raise NotImplementedError(f"{type(self).__name__} does not implement cdf()")

    def ppf(self, q: float) -> float:
        """Percent-point function (inverse CDF / quantile function)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement ppf()")
