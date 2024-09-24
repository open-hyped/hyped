import math
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from hyped.common.utils import (
    chdir,
    deep_equal,
    dict_of_lists_to_list_of_dicts,
    is_package_installed,
    list_of_dicts_to_dict_of_lists,
    time_weighted_ema,
)


def test_list_of_dicts_to_dict_of_lists():
    # Test case 1: Basic functionality
    dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    keys = ["a", "b"]
    expected = {"a": [1, 3], "b": [2, 4]}
    assert list_of_dicts_to_dict_of_lists(dicts, keys) == expected

    # Test case 2: Default value when a key is missing
    dicts = [{"a": 1}, {"b": 4}]
    keys = ["a", "b"]
    expected = {"a": [1, None], "b": [None, 4]}
    assert list_of_dicts_to_dict_of_lists(dicts, keys) == expected

    # Test case 3: Custom default value
    dicts = [{"a": 1}, {"b": 4}]
    keys = ["a", "b"]
    expected = {"a": [1, 0], "b": [0, 4]}
    assert list_of_dicts_to_dict_of_lists(dicts, keys, default=0) == expected


def test_dict_of_lists_to_list_of_dicts():
    # Test case 1: Basic functionality
    dict_of_lists = {"a": [1, 3], "b": [2, 4]}
    expected = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert dict_of_lists_to_list_of_dicts(dict_of_lists) == expected

    # Test case 2: Empty dictionary
    dict_of_lists = {}
    expected = []
    assert dict_of_lists_to_list_of_dicts(dict_of_lists) == expected

    # Test case 3: Lists of different sizes should raise an AssertionError
    dict_of_lists = {"a": [1, 3], "b": [2]}
    with pytest.raises(AssertionError):
        dict_of_lists_to_list_of_dicts(dict_of_lists)


def test_basic_types():
    # Test with basic types
    assert deep_equal(10, 10)
    assert not deep_equal(10, 20)
    assert deep_equal("hello", "hello")
    assert not deep_equal("hello", "world")
    assert deep_equal(3.14, 3.14)
    assert not deep_equal(3.14, 2.71)


def test_lists_and_tuples():
    # Test with lists
    assert deep_equal([1, 2, 3], [1, 2, 3])
    assert not deep_equal([1, 2, 3], [3, 2, 1])
    assert deep_equal((1, 2, 3), (1, 2, 3))
    assert not deep_equal((1, 2, 3), (3, 2, 1))
    # Test with different lenghts
    assert not deep_equal((1, 2, 3), (1, 2))

    # Test with nested lists
    assert deep_equal([1, [2, 3], 4], [1, [2, 3], 4])
    assert not deep_equal([1, [2, 3], 4], [1, [2, 4], 3])


def test_dictionaries():
    # Test with dictionaries
    assert deep_equal({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert not deep_equal({"a": 1, "b": 2}, {"a": 2, "b": 1})
    # Test with different keys
    assert not deep_equal({"a": 1, "b": 2}, {"c": 2, "d": 1})

    # Test with nested dictionaries
    assert deep_equal({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
    assert not deep_equal({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 3}})


def test_numpy_arrays():
    # Test with numpy arrays
    assert deep_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert not deep_equal(np.array([1, 2, 3]), np.array([3, 2, 1]))

    # Test with nested numpy arrays
    assert deep_equal(
        [np.array([1, 2]), np.array([3, 4])],
        [np.array([1, 2]), np.array([3, 4])],
    )
    assert not deep_equal(
        [np.array([1, 2]), np.array([3, 4])],
        [np.array([1, 2]), np.array([4, 3])],
    )


def test_mixed_structures():
    # Test with mixed structures
    assert deep_equal(
        {"a": [1, 2, {"b": np.array([3, 4])}]},
        {"a": [1, 2, {"b": np.array([3, 4])}]},
    )
    assert not deep_equal(
        {"a": [1, 2, {"b": np.array([3, 4])}]},
        {"a": [1, 2, {"b": np.array([4, 3])}]},
    )


def test_chdir():
    """Test the chdir context manager."""
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir1:
        orig_dir = os.getcwd()

        # Test the chdir context manager
        with chdir(temp_dir1):
            assert os.getcwd().endswith(os.path.abspath(temp_dir1))  # Should be in temp_dir2

        # After exiting the context, should be back to temp_dir1
        assert os.getcwd().endswith(os.path.abspath(orig_dir))


def _test_chdir_exception():
    """Test the chdir context manager with an exception."""
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir1:
        orig_dir = os.getcwd()

        # Test the chdir context manager with an exception
        with pytest.raises(RuntimeError):
            with chdir(temp_dir1):
                assert os.getcwd().endswith(os.path.abspath(temp_dir1))  # Should be in temp_dir2
                raise RuntimeError("Test exception")  # Raise an exception

        # After exiting the context, should be back to temp_dir1
        assert os.getcwd().endswith(os.path.abspath(orig_dir))


def test_calculate_time_weighted_ema():
    measurements = {
        1695419000: 100,  # Timestamp: 1695419000, Measurement: 100
        1695419100: 150,  # 100 seconds later
        1695419800: 200,  # 700 seconds later
        1695420400: 250,  # 600 seconds later
    }

    actual_ema = time_weighted_ema(measurements, decay_rate=0.001)

    # Validate the result against expected values
    assert actual_ema is not None
    assert isinstance(actual_ema, float)
    # Verify the calculated EMA
    assert math.isclose(actual_ema, 215.00310219329043, rel_tol=1e-5)


def test_is_package_installed_existing_package():
    # Mock `importlib.util.find_spec` to return a mock object for an existing package
    with patch("importlib.util.find_spec", return_value=object()):
        # Test for a package that exists (mocked)
        assert is_package_installed("existing_package") is True


def test_is_package_installed_non_existing_package():
    # Mock `importlib.util.find_spec` to return None for a non-existing package
    with patch("importlib.util.find_spec", return_value=None):
        # Test for a package that does not exist
        assert is_package_installed("non_existing_package") is False


def test_is_package_installed_real_package():
    # Test with a package that is actually installed (e.g., 'pytest')
    assert is_package_installed("pytest") is True


def test_is_package_installed_real_non_existing_package():
    # Test with a package that is very unlikely to exist
    assert is_package_installed("some_non_existing_random_package") is False
