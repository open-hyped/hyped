import numpy as np
import pytest

from hyped.common.utils import (
    deep_equal,
    dict_of_lists_to_list_of_dicts,
    list_of_dicts_to_dict_of_lists,
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
