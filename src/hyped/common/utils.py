"""A Collection of utility functions used throughout the project."""

import importlib.util
import os
import sys
from contextlib import contextmanager
from functools import cache
from typing import Any, Generator

import numpy as np


# TODO: potential legacy code
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
def tmp_setattr(instance: object, attribute: str, value: Any) -> Generator[None, None, None]:
    # Save the original value
    original_value = getattr(instance, attribute, None)
    has_original_value = hasattr(instance, attribute)

    # Set the temporary value
    setattr(instance, attribute, value)
    try:
        yield
    finally:
        # Restore the original value
        if has_original_value:
            setattr(instance, attribute, original_value)
        else:
            delattr(instance, attribute)  # Clean up if there was no original value


@contextmanager
def chdir(new_dir: str) -> Generator[None, None, None]:
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
    """Check if a package with the given name is installed.

    Args:
        package_name (str): The name of the package.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


def is_python_version_less_than(major, minor: int = 0, micro: int = 0):
    """Check if the current Python version is less than the provided version.

    Args:
        major (int): Major version to compare.
        minor (int): Minor version to compare. Defaults to 0.
        micro (int): Micro version to compare. Defaults to 0.

    Returns:
        bool: True if the current Python version is less than the provided version.
    """
    current_version = sys.version_info
    target_version = (major, minor, micro)
    return current_version < target_version
