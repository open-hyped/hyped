"""Module for various data aggregators in data processing workflows.

This module provides a comprehensive collection of data aggregators designed to
manage the aggregation of different data types and perform a wide range of data
aggregation operations. Data aggregators are essential components in a data flow
graph, acting as nodes that implement specific, modular aggregation logic.
"""

from typing import TYPE_CHECKING

__all__ = [
    "MultiLabelConfusionMatrix",
]

if TYPE_CHECKING:
    from .confusion.mcm import MultiLabelConfusionMatrix

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {"MultiLabelConfusionMatrix": "hyped.aggregators.confusion.mcm"}

    sys.modules[__name__] = LazyModule(
        __name__, __doc__, globals()["__file__"], __spec__, _lazy_imports
    )
