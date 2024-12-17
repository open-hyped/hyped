"""Module providing the base test for nodes in a data processing pipeline."""

from hyped.common.utils import is_package_installed

if not is_package_installed("pytest"):
    raise EnvironmentError("PyTest not installed!")
