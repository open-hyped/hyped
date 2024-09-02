"""Module for operator processors in data processing workflows.

This module defines various operator processors that perform specific
operations within a data flow graph. Operator processors are essential
building blocks in constructing complex data processing pipelines,
enabling efficient and modular handling of data transformations.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .binary import (
        Add,
        Equals,
        FloorDiv,
        GreaterThan,
        GreaterThanOrEqual,
        LessThan,
        LessThanOrEqual,
        LogicalAnd,
        LogicalOr,
        LogicalXOr,
        Mod,
        Mul,
        NotEquals,
        Pow,
        Sub,
        TrueDiv,
    )
    from .collect import CollectFeatures
    from .noop import NoOp
    from .sequence.access import (
        BooleanIndexing,
        SequenceGetItem,
        SequenceSetItem,
    )
    from .sequence.multi import SequenceChain, SequenceZip
    from .sequence.query import (
        SequenceContains,
        SequenceCountOf,
        SequenceIndexOf,
    )
    from .sequence.reduce import SequenceLength, SequenceMean, SequenceSum
    from .unary import Abs, BooleanInvert, Invert, Neg

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        # binary
        "Add": "hyped.processors._ops.binary",
        "Equals": "hyped.processors._ops.binary",
        "FloorDiv": "hyped.processors._ops.binary",
        "GreaterThan": "hyped.processors._ops.binary",
        "GreaterThanOrEqual": "hyped.processors._ops.binary",
        "LessThan": "hyped.processors._ops.binary",
        "LessThanOrEqual": "hyped.processors._ops.binary",
        "LogicalAnd": "hyped.processors._ops.binary",
        "LogicalOr": "hyped.processors._ops.binary",
        "LogicalXOr": "hyped.processors._ops.binary",
        "Mod": "hyped.processors._ops.binary",
        "Mul": "hyped.processors._ops.binary",
        "NotEquals": "hyped.processors._ops.binary",
        "Pow": "hyped.processors._ops.binary",
        "Sub": "hyped.processors._ops.binary",
        "TrueDiv": "hyped.processors._ops.binary",
        # collect
        "CollectFeatures": "hyped.processors._ops.collect",
        # noop
        "NoOp": "hyped.processors._ops.noop",
        # sequence
        "BooleanIndexing": "hyped.processors._ops.sequence.access",
        "SequenceGetItem": "hyped.processors._ops.sequence.access",
        "SequenceSetItem": "hyped.processors._ops.sequence.access",
        "SequenceChain": "hyped.processors._ops.sequence.multi",
        "SequenceZip": "hyped.processors._ops.sequence.multi",
        "SequenceContains": "hyped.processors._ops.sequence.query",
        "SequenceCountOf": "hyped.processors._ops.sequence.query",
        "SequenceIndexOf": "hyped.processors._ops.sequence.query",
        "SequenceLength": "hyped.processors._ops.sequence.reduce",
        "SequenceMean": "hyped.processors._ops.sequence.reduce",
        "SequenceSum": "hyped.processors._ops.sequence.reduce",
        # unary
        "Abs": "hyped.processors._ops.unary",
        "BooleanInvert": "hyped.processors._ops.unary",
        "Invert": "hyped.processors._ops.unary",
        "Neg": "hyped.processors._ops.unary",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
    )
