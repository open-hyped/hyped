"""Provides operator-like data aggregators for performing common statistical operations.

This module includes data aggregators that perform basic statistical operations such as
:code:`sum`, :code:`mean`, and :code:`standard deviation` on input features. These aggregators are
designed to be used within data processing pipelines to compute dataset-wide statistics. Each
aggregator follows a standard interface for initialization, extraction of values from batches,
and updating the aggregated results.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mean import MeanAggregator
    from .sum import SumAggregator

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        "MeanAggregator": "hyped.aggregators._ops.mean",
        "SumAggregator": "hyped.aggregators._ops.sum",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
    )
