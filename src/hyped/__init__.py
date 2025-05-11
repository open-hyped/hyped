"""A comprehensive framework for complex data processing pipelines.

This package provides a comprehensive framework for constructing, managing,
and executing complex data processing pipelines. The framework is designed
to be modular and flexible, allowing users to define data flows to handle
a wide variety of data processing tasks.
"""

import os

from .__version__ import __version__, __version_tuple__  # noqa: F401

# isort: off
# keep logging setup at beginning of the file
# to run the setup hook first
from hyped.common.logging import setup_logging

# setup logging
setup_logging(level=os.getenv("LOG_LEVEL", "WARNING").upper(), log_file=os.getenv("LOG_FILE", None))

# isort: on

__all__ = [
    # modules
    "core",
    "typing",
    "ops",
    # core
    "DataFlow",
    "plot_data_flow",
]

from . import core, ops, typing  # noqa: E402
from .core.flow import DataFlow, plot_data_flow  # noqa: E402
