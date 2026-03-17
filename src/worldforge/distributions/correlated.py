"""Correlated multivariate distributions via Gaussian copula."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from worldforge.core.exceptions import DistributionError
from worldforge.distributions.base import Distribution


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy required)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _normal_cdf_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF."""
    return 0.5 * (1 + np.vectorize(math.erf)(arr / math.sqrt(2)))


class CorrelatedDistributions:
    """
    Generate correlated samples from multiple distributions using a Gaussian copula.

    Steps:
    1. Draw correlated standard-normal samples via Cholesky decomposition.
    2. Map to [0,1] via the standard normal CDF.
    3. Apply each distribution's inverse CDF (ppf) to obtain marginal samples.

    Requires all component distributions to implement ppf().

    Example::

        price, qty = CorrelatedDistributions(
            distributions=[LogNormal(4, 0.5), Poisson(10)],
            correlation=-0.7,
        ).sample(rng)
    """

    def __init__(
        self,
        distributions: Sequence[Distribution],
        correlation,
    ) -> None:
        n = len(distributions)
        if n < 2:
            raise DistributionError("Need at least 2 distributions")

        self.distributions = list(distributions)

        # Build correlation matrix
        if isinstance(correlation, (int, float)):
            # Uniform pairwise correlation
            corr_matrix = np.full((n, n), float(correlation))
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            corr_matrix = np.array(correlation, dtype=float)
            if corr_matrix.shape != (n, n):
                raise DistributionError(
                    f"correlation matrix must be {n}x{n}, got {corr_matrix.shape}"
                )

        # Validate positive semi-definiteness via Cholesky
        try:
            self._chol = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            raise DistributionError(
                "Correlation matrix is not positive definite. "
                "Check that all correlations are in (-1, 1) and the matrix is valid."
            )

    def sample(self, rng: np.random.Generator) -> tuple:
        """Return a tuple of correlated samples, one per distribution."""
        n = len(self.distributions)
        # Independent standard normals
        z = rng.standard_normal(n)
        # Introduce correlation via Cholesky
        correlated_z = self._chol @ z
        # Map to uniform [0,1] via standard normal CDF
        u = _normal_cdf_array(correlated_z)
        # Inverse CDF transform for each distribution
        return tuple(dist.ppf(float(np.clip(u[i], 1e-9, 1 - 1e-9))) for i, dist in enumerate(self.distributions))

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return array of shape (n, k) where k = number of distributions."""
        k = len(self.distributions)
        # (k, n) matrix of independent standard normals
        z = rng.standard_normal((k, n))
        # Introduce correlation: (k, n)
        correlated_z = self._chol @ z
        # Map to uniform [0,1]
        u = _normal_cdf_array(correlated_z)  # (k, n)
        u = np.clip(u, 1e-9, 1 - 1e-9)
        # Apply inverse CDF for each distribution
        result = np.empty((n, k))
        for i, dist in enumerate(self.distributions):
            result[:, i] = np.array([dist.ppf(float(q)) for q in u[i]])
        return result

    def __repr__(self) -> str:
        return f"CorrelatedDistributions(n={len(self.distributions)})"
