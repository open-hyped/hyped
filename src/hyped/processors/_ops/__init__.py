"""Module for operator processors in data processing workflows.

This module defines various operator processors that perform specific
operations within a data flow graph. Operator processors are essential
building blocks in constructing complex data processing pipelines,
enabling efficient and modular handling of data transformations.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import binary, collect, noop, sequence, unary

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_modules = {
        "unary": "hyped.processors._ops.unary",
        "binary": "hyped.processors._ops.binary",
        "collect": "hyped.processors._ops.collect",
        "noop": "hyped.processors._ops.noop",
        "sequence": "hyped.processors._ops.sequence",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_modules=_lazy_modules,
    )
