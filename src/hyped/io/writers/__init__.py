"""Provides dataset writers for saving processed data.

This module offers various dataset writers for saving processed data in different formats.
It includes a base dataset consumer and specific implementations like Arrow and JSON writers.
These writers handle the logic for serializing and storing datasets efficiently.
"""

from typing import TYPE_CHECKING

__all__ = (
    "DatasetConsumer",
    "ShardingStrategy",
    "ArrowDatasetWriter",
    "JsonDatasetWriter",
)

if TYPE_CHECKING:
    from .arrow import ArrowDatasetWriter
    from .base import DatasetConsumer, ShardingStrategy
    from .json import JsonDatasetWriter

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        "DatasetConsumer": "hyped.io.writers.base",
        "ShardingStrategy": "hyped.io.writers.base",
        "ArrowDatasetWriter": "hyped.io.writers.arrow",
        "JsonDatasetWriter": "hyped.io.writers.json",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
    )
