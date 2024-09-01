"""Package for sequence data processing operations.

This package includes modules that implement various data processors for
sequence features, such as item access, item operations, query operations,
and reduction operations.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import access, multi, query, reduce

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_modules = {
        "access": "hyped.processors._ops.sequence.access",
        "multi": "hyped.processors._ops.sequence.multi",
        "query": "hyped.processors._ops.sequence.query",
        "reduce": "hyped.processors._ops.sequence.reduce",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_modules=_lazy_modules,
    )
