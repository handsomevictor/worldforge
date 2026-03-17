"""Mixture distribution: weighted combination of component distributions."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from worldforge.core.exceptions import DistributionError
from worldforge.distributions.base import Distribution


class MixtureDistribution(Distribution):
    """
    Finite mixture: sample a component by weight, then sample from it.

    Example (bimodal spending)::

        MixtureDistribution(
            components=[Normal(50, 10), Normal(500, 100)],
            weights=[0.8, 0.2],
        )
    """

    def __init__(
        self,
        components: Sequence[Distribution],
        weights: Sequence[float],
    ) -> None:
        if len(components) == 0:
            raise DistributionError("components must not be empty")
        if len(components) != len(weights):
            raise DistributionError("components and weights must have the same length")
        if any(w < 0 for w in weights):
            raise DistributionError("all weights must be >= 0")
        total = sum(weights)
        if total == 0:
            raise DistributionError("weights must not all be zero")
        self.components = list(components)
        self._probs = np.array([w / total for w in weights])

    def _pick_component(self, rng: np.random.Generator) -> Distribution:
        idx = int(rng.choice(len(self.components), p=self._probs))
        return self.components[idx]

    def sample(self, rng: np.random.Generator) -> float:
        return self._pick_component(rng).sample(rng)

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Assign each sample to a component, then draw from that component
        component_indices = rng.choice(len(self.components), size=n, p=self._probs)
        result = np.empty(n)
        for idx, component in enumerate(self.components):
            mask = component_indices == idx
            count = int(mask.sum())
            if count > 0:
                result[mask] = component.sample_batch(count, rng)
        return result

    def mean(self) -> float:
        return float(sum(p * c.mean() for p, c in zip(self._probs, self.components)))

    def std(self) -> float:
        # Law of total variance: Var(X) = E[Var(X|Z)] + Var(E[X|Z])
        mu = self.mean()
        ev = float(sum(p * c.std() ** 2 for p, c in zip(self._probs, self.components)))
        ve = float(sum(p * (c.mean() - mu) ** 2 for p, c in zip(self._probs, self.components)))
        return math.sqrt(ev + ve)

    def __repr__(self) -> str:
        return f"MixtureDistribution(n_components={len(self.components)})"
