"""Module for various data augmenters in data processing workflows.

This module provides a comprehensive collection of data augmenters designed to
enhance datasets by generating new samples or filtering existing ones. Data
augmenters are key components in a data flow graph, acting as nodes that
introduce flexibility in dataset size and diversity, and are seamlessly integrated
into the workflow to support a wide range of data augmentation operations.
"""

from typing import TYPE_CHECKING

__all__ = [
    "ChunkSequence",
]

if TYPE_CHECKING:
    from .chunk import ChunkSequence

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {"ChunkSequence": "hyped.augmenters.chunk"}

    sys.modules[__name__] = LazyModule(
        __name__, __doc__, globals()["__file__"], __spec__, _lazy_imports
    )
