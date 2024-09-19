"""A Collection of utility functions used throughout the project."""

import importlib.util
import os
from contextlib import contextmanager
from functools import cache
from queue import Empty, Queue
from typing import Any, Hashable, Iterable

import numpy as np


def list_of_dicts_to_dict_of_lists(
    dicts: list[dict[Hashable, Any]], keys: list[Hashable], default: Any = None
) -> dict[Hashable, list[Any]]:
    """Convert a list-of-dicts to a dict-of-lists.

    Args:
        dicts (list[dict[Hashable, Any]]): List of dictionaries to convert.
        keys (list[Hashable]): Keys to extract from the dictionaries.
        default (Any): Default value to fill in in case the dictionary doesn't contain a key.

    Returns:
        dict[Hashable, list[Any]]: Dictionary of lists containing values collected from the input.
    """
    return {key: [d.get(key, default) for d in dicts] for key in keys}


def dict_of_lists_to_list_of_dicts(
    dict_of_lists: dict[Hashable, list[Any]]
) -> list[dict[Hashable, Any]]:
    """Convert a dict-of-lists to a list-of-dicts.

    Args:
        dict_of_lists (dict[Hashable, list[Any]]): Dictionaries of lists
            to convert.

    Returns:
        list[dict[Hashable, Any]]: List of dictionaries containing values
            collected from the input.

    Raises:
        AssertionError: If the lists in the dictionary have varying sizes.
    """
    keys = dict_of_lists.keys()

    # make sure all lists are of the same size
    assert all(
        len(vals) == len(dict_of_lists[next(iter(keys))]) for vals in dict_of_lists.values()
    ), "All lists must have the same length."

    return [dict(zip(keys, vals)) for vals in zip(*dict_of_lists.values())]


def deep_equal(obj1: Any, obj2: Any) -> bool:
    """Recursively checks if two objects (which may be nested) are equal.

    Handles basic types, dictionaries, lists, tuples, and numpy arrays.

    Args:
        obj1: The first object to compare.
        obj2: The second object to compare.

    Returns:
        bool: True if obj1 and obj2 are equal, False otherwise.
    """
    # If both objects are numpy arrays, use np.array_equal
    if isinstance(obj1, np.ndarray) or isinstance(obj2, np.ndarray):
        return np.array_equal(obj1, obj2)

    # If both objects are dictionaries, compare keys and values recursively
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_equal(obj1[k], obj2[k]) for k in obj1)

    # If both objects are lists or tuples, compare elements recursively
    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(deep_equal(i1, i2) for i1, i2 in zip(obj1, obj2))

    # For all other types, use standard equality
    return obj1 == obj2


@contextmanager
def chdir(new_dir: str):
    """Context manager for temporarily changing the working directory.

    Args:
        new_dir (str): The directory to change to temporarily.

    Yields:
        None
    """
    original_dir = os.getcwd()  # Save the current working directory
    os.chdir(new_dir)  # Change to the new directory
    try:
        yield  # Yield control back to the caller
    finally:
        os.chdir(original_dir)  # Restore the original directory


@cache
def is_package_installed(package_name: str) -> bool:
    """
    Check if a package with the given name is installed.

    Args:
        package_name (str): The name of the package.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


class QueueIterator:
    """An iterator for consuming items from a queue.

    This iterator retrieves items from a queue with an optional timeout and can
    terminate iteration upon encountering a sentinel value.
    """

    def __init__(self, queue: Queue, sentinel: Any = None, timeout: None | float = None) -> None:
        """Initialize the iterator.

        Args:
            queue (Queue): The queue to iterate through.
            sentinel (Any): A special value to signal the end of the queue.
            timeout (float): Timeout for waiting on queue items.
        """
        self.queue = queue
        self.sentinel = sentinel
        self.timeout = timeout

    def __iter__(self) -> Iterable[Any]:
        """Return the iterator object itself.

        Returns:
            Iterable[Any]: The iterator object itself.
        """
        return self

    def __next__(self) -> Any:
        """Retrieve the next item from the queue.

        Returns:
            Any: The next item from the queue.

        Raises:
            StopIteration: If the sentinel value is encountered or if the queue is empty.
        """
        try:
            # Get the next item from the queue with a timeout
            item = self.queue.get(timeout=self.timeout)

            # If the sentinel value is encountered, raise StopIteration to end iteration
            if item == self.sentinel:
                raise StopIteration

            return item
        except Empty:
            raise StopIteration


class StoppableIterator(object):
    """An iterator that can be stopped manually.

    This iterator allows manual interruption of iteration by calling the :func:`stop`
    method, which causes the iterator to stop yielding items.
    """

    def __init__(self, iterable: Iterable[Any]) -> None:
        """Initialize the stoppable iterator.

        Args:
            iterable (Iterable[Any]): The iterable to iterate over.
        """
        self.iterable = iter(iterable)
        self.stopped = False

    def stop(self) -> None:
        """Stop the iteration.

        Sets the stopped flag to True, causing the iterator to stop yielding items.
        """
        self.stopped = True

    def __iter__(self) -> Iterable[Any]:
        """Return the iterator object itself.

        Returns:
            Iterable[Any]: The iterator object itself.
        """
        return self

    def __next__(self) -> Any:
        """Retrieve the next item from the iterable.

        Returns:
            Any: The next item from the iterable.

        Raises:
            StopIteration: If iteration is stopped or if the iterable is exhausted.
        """
        if self.stopped:
            raise StopIteration

        return next(self.iterable)
