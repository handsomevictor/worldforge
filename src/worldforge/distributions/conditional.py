"""Conditional distribution: dispatch to different distributions based on a key."""
from __future__ import annotations

from typing import Callable, Any

import numpy as np

from worldforge.core.exceptions import DistributionError
from worldforge.distributions.base import Distribution


class ConditionalDistribution(Distribution):
    """
    Dispatches to a specific sub-distribution based on a condition evaluated on
    an agent (or any callable that returns a hashable key).

    Example::

        bonus = field(ConditionalDistribution(
            condition=lambda agent: agent.tier,
            mapping={
                "free":       Uniform(0, 0),
                "pro":        Normal(500, 100),
                "enterprise": Normal(5000, 1000),
            }
        ))

    When `sample(rng, context=agent)` is called, `condition(agent)` is evaluated
    to select the appropriate sub-distribution.
    """

    def __init__(
        self,
        condition: Callable[[Any], Any],
        mapping: dict[Any, Distribution],
        default: Distribution | None = None,
    ) -> None:
        if not callable(condition):
            raise DistributionError("condition must be callable")
        if not mapping:
            raise DistributionError("mapping must not be empty")
        self.condition = condition
        self.mapping = dict(mapping)
        self.default = default

    def _resolve(self, context=None) -> Distribution:
        if context is None:
            # No context: return the first distribution as a fallback
            return next(iter(self.mapping.values()))
        key = self.condition(context)
        if key in self.mapping:
            return self.mapping[key]
        if self.default is not None:
            return self.default
        raise DistributionError(
            f"ConditionalDistribution: key {key!r} not in mapping and no default set. "
            f"Available keys: {list(self.mapping.keys())}"
        )

    def sample(self, rng: np.random.Generator, context=None) -> float:
        return self._resolve(context).sample(rng)

    def sample_batch(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return self._resolve(context).sample_batch(n, rng)

    def mean(self) -> float:
        raise NotImplementedError(
            "ConditionalDistribution.mean() requires a context. "
            "Call mean() on the resolved sub-distribution."
        )

    def __repr__(self) -> str:
        return f"ConditionalDistribution(keys={list(self.mapping.keys())})"
