"""A comprehensive framework for complex data processing pipelines.

This package provides a comprehensive framework for constructing, managing,
and executing complex data processing pipelines. The framework is designed
to be modular and flexible, allowing users to define data flows to handle
a wide variety of data processing tasks.
"""

from .__version__ import __version__, __version_tuple__  # noqa: F401

__all__ = [
    # modules
    "io",
    "ops",
    "nodes",
    "core",
    # core
    "DataFlow",
]

from . import core, io, nodes, ops
from .core.flow import DataFlow
