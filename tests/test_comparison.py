from __future__ import annotations

import numpy as np
import pytest

from comparison.methods import METHOD_REGISTRY, get_method
from comparison.methods.base import OptimizerResult


def test_optimizer_result_creation():
    result = OptimizerResult(
        x_optimal=np.array([1.0, 2.0]),
        f_optimal=3.0,
        iterations=10,
    )
    assert result.f_optimal == 3.0
    assert result.iterations == 10
    assert result.x_optimal.shape == (2,)


def test_registry_starts_empty():
    assert isinstance(METHOD_REGISTRY, dict)


def test_get_method_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown optimizer method"):
        get_method("nonexistent_method_xyz")
