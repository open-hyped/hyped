"""A Collection of utility functions used throughout the project."""

import importlib.util
import os
from contextlib import contextmanager
from functools import cache, reduce
from typing import Any, Callable, Hashable

import numpy as np


class compose(object):
    """Composes an arbitrary number of functions into a single function.

    The composed function applies the input functions from right to left (i.e.,
    the last function in the list is applied first, and the first function is applied last).

    Example:
        If :code:`compose(f, g, h)` is called with input :code:`x`, it returns :code:`f(g(h(x)))`.

    Args:
        *functions (Callable[[Any], Any]): An arbitrary number of functions to compose.
            Each function must accept the output of the subsequent function (or the initial input).
            Must contain at least one function.

    Returns:
        Callable[[Any], Any]: A function that applies the composed functions sequentially from
        right to left.
    """

    def __init__(self, *functions: Callable[[Any], Any]) -> None:
        """Initialize the :class:`compose` object with the provided functions.

        Args:
            *functions (Callable[[Any], Any]): Functions to be composed.
        """
        assert len(functions) > 0, "At least one function must be provided"
        self._functions = tuple(reversed(functions))

    def __call__(self, x: Any) -> Any:
        """Applies the composed functions to the input.

        Args:
            x (Any): The initial input to be passed through the composed functions.

        Returns:
            Any: The result of applying the composed functions sequentially.
        """
        return reduce(lambda x, f: f(x), self._functions, x)


class run_all(object):
    """Runs an arbitrary number of functions sequentially with the same arguments.

    Each function in the list is called with the provided arguments, and their execution
    order is from first to last. No function's output is used as input for the next.

    Example:
        If :code:`run_all(f, g, h)` is called with arguments :code:`x, y`, it executes:

        .. code-block:: python

            f(x, y)
            g(x, y)
            h(x, y)

    Args:
        *functions (Callable[[Any], Any]): An arbitrary number of functions to be run
            sequentially. Each function must accept the same arguments.

    Returns:
        None
    """

    def __init__(self, *functions: Callable[[Any], Any]) -> None:
        """Initialize the :class:`run_all` object with the provided functions.

        Args:
            *functions (Callable[[Any], Any]): Functions to be executed sequentially.
        """
        self._functions = functions

    def __call__(self, *args, **kwargs) -> None:
        """Executes all functions with the provided arguments and keyword arguments.

        Args:
            *args (Any): Positional arguments to be passed to each function.
            **kwargs (Any): Keyword arguments to be passed to each function.

        Returns:
            None
        """
        for f in self._functions:
            f(*args, **kwargs)


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
