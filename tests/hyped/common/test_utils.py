from queue import Queue
from unittest.mock import patch

import numpy as np
import pytest

from hyped.common.utils import (
    QueueIterator,
    StoppableIterator,
    deep_equal,
    dict_of_lists_to_list_of_dicts,
    is_package_installed,
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


class TestQueueIterator:
    def test_queue_iterator_basic(self):
        q = Queue()
        items = [1, 2, 3, 4]

        # Put items in the queue
        for item in items:
            q.put(item)
        q.put(None)  # Sentinel value

        iterator = QueueIterator(q, sentinel=None)

        # Ensure that all items are returned in order
        assert list(iterator) == items

    def test_queue_iterator_empty_queue(self):
        q = Queue()
        iterator = QueueIterator(q, sentinel=None, timeout=0.1)

        # Ensure that StopIteration is raised when queue is empty
        with pytest.raises(StopIteration):
            next(iterator)

    def test_queue_iterator_with_sentinel(self):
        q = Queue()
        items = [1, 2, 3, "stop", 5]

        # Put items in the queue
        for item in items:
            q.put(item)

        # Sentinel value is "stop"
        iterator = QueueIterator(q, sentinel="stop")

        # Ensure items before sentinel are returned
        assert list(iterator) == [1, 2, 3]

    def test_queue_iterator_with_timeout(self):
        q = Queue()

        # No items in the queue, should raise StopIteration after timeout
        iterator = QueueIterator(q, timeout=0.1)
        with pytest.raises(StopIteration):
            next(iterator)


class TestStoppableIterator:
    def test_stoppable_iterator_basic(self):
        iterable = [1, 2, 3, 4, 5]
        iterator = StoppableIterator(iterable)

        # Ensure all items are returned in order
        assert list(iterator) == iterable

    def test_stoppable_iterator_stop(self):
        iterable = [1, 2, 3, 4, 5]
        iterator = StoppableIterator(iterable)

        # Retrieve first two items
        assert next(iterator) == 1
        assert next(iterator) == 2

        # Stop the iterator
        iterator.stop()

        # Ensure StopIteration is raised after stop is called
        with pytest.raises(StopIteration):
            next(iterator)

    def test_stoppable_iterator_exhausted(self):
        iterable = [1, 2]
        iterator = StoppableIterator(iterable)

        # Exhaust the iterator
        assert next(iterator) == 1
        assert next(iterator) == 2

        # Ensure StopIteration is raised after all items are exhausted
        with pytest.raises(StopIteration):
            next(iterator)
