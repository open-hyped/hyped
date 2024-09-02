"""A Collection of utility functions used throughout the project."""

from typing import Any, Hashable

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
