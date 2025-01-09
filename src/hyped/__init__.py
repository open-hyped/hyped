"""A comprehensive framework for complex data processing pipelines.

This package provides a comprehensive framework for constructing, managing,
and executing complex data processing pipelines. The framework is designed
to be modular and flexible, allowing users to define data flows to handle
a wide variety of data processing tasks.
"""

from .__version__ import __version__, __version_tuple__  # noqa: F401

# isort: off
# keep logging setup at beginning of the file
# to run the setup hook first
from .logging.setup import setup_logging

# isort: on

__all__ = [
    # modules
    "core",
    "typing",
    "ops",
    # core
    "DataFlow",
    "plot_data_flow",
    # logging
    "setup_logging",
]

from . import core, ops, typing
from .core.flow import DataFlow, plot_data_flow
