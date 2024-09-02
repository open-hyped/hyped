"""Dataset Writers."""

from typing import TYPE_CHECKING

__all__ = [
    "CsvDatasetWriter",
    "JsonDatasetWriter",
]

if TYPE_CHECKING:
    from .csv import CsvDatasetWriter
    from .json import JsonDatasetWriter

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        "CsvDatasetWriter": "hyped.io.writers.csv",
        "JsonDatasetWriter": "hyped.io.writers.json",
    }

    sys.modules[__name__] = LazyModule(
        __name__, __doc__, globals()["__file__"], __spec__, _lazy_imports
    )
