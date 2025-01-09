"""A Collection of utility functions used throughout the project."""

import importlib.util
import sys
from contextlib import contextmanager
from functools import cache
from typing import Any, Generator


@contextmanager
def tmp_setattr(instance: object, attribute: str, value: Any) -> Generator[None, None, None]:
    """Temporarily set an attribute on an object and restore its original value afterward.

    This context manager temporarily sets the specified attribute on the given object to a new
    value. Upon exiting the context, the original value is restored. If the attribute did not
    exist prior to the context, it will be removed when the context exits.

    Args:
        instance (object): The object on which the attribute will be temporarily set.
        attribute (str): The name of the attribute to be set.
        value (Any): The temporary value to assign to the attribute.

    Yields:
        None: Yields control to the context block.

    Raises:
        AttributeError: If the attribute cannot be set or removed on the given object.
    """
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
