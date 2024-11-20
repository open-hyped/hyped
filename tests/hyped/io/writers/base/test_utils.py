import importlib
import math
from unittest.mock import MagicMock, patch

import pytest

import hyped.io.writers.base.utils
from hyped.io.writers.base.utils import EMA, Compose, RunAll, TimeWeightedEMA, ith_entries


@patch("hyped.common.utils.is_python_version_less_than", return_value=(3, 11))
def test_batched(mock_is_python_version_less_than: MagicMock) -> None:
    # reload module to make sure to use the <3.12 implementation
    importlib.reload(hyped.io.writers.base.utils)
    from hyped.io.writers.base.utils import batched

    # check batches
    assert list(batched(range(100), 10)) == [tuple(range(i, i + 10)) for i in range(0, 100, 10)]

    # batch size to small
    with pytest.raises(ValueError):
        next(batched(range(10), 0))


def test_ith_entries() -> None:
    it = [(i, i + 1) for i in range(10)]

    assert list(ith_entries(it, 0)) == list(range(10))
    assert list(ith_entries(it, 1)) == list(range(1, 11))


def test_compose() -> None:
    f, g, x = MagicMock(), MagicMock(), MagicMock()
    assert Compose(f, g)(x) == f(g(x))


def test_run_all() -> None:
    f, g, x = MagicMock(), MagicMock(), MagicMock()
    RunAll(f, g)(x)
    # make sure functions are called
    f.assert_called_once_with(x)
    g.assert_called_once_with(x)


def test_calculate_ema():
    measurements = [100, 150, 200, 250]  # Measurements taken over time

    ema = EMA(smoothing=0.3)

    for value in measurements:
        ema(value)

    # Validate the result against expected values
    assert ema.value is not None
    assert isinstance(ema.value, float)
    # Verify the calculated EMA against an expected value
    assert math.isclose(ema.value, 228.12212133175416, rel_tol=1e-5)


def test_calculate_time_weighted_ema():
    measurements = {
        1695419000: 100,  # Timestamp: 1695419000, Measurement: 100
        1695419100: 150,  # 100 seconds later
        1695419800: 200,  # 700 seconds later
        1695420400: 250,  # 600 seconds later
    }

    ema = TimeWeightedEMA(decay_rate=0.001)

    for timestamp, value in measurements.items():
        ema(timestamp, value)

    # Validate the result against expected values
    assert ema.value is not None
    assert isinstance(ema.value, float)
    # Verify the calculated EMA
    assert math.isclose(ema.value, 215.00310219329043, rel_tol=1e-5)
