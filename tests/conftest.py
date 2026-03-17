"""Shared pytest fixtures."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded RNG for deterministic tests."""
    return np.random.default_rng(42)
