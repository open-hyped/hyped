"""A comprehensive framework for complex data processing pipelines.

This package provides a comprehensive framework for constructing, managing,
and executing complex data processing pipelines. The framework is designed
to be modular and flexible, allowing users to define data flows to handle
a wide variety of data processing tasks.
"""

from .__version__ import __version__, __version_tuple__  # noqa: F401

__all__ = [
    # modules
    "core",
    "typing",
    # core
    "DataFlow",
    "plot_data_flow",
]

from . import core, typing
from .core.flow import DataFlow, plot_data_flow
