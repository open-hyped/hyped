"""Module for operator processors in data processing workflows.

This module defines various operator processors that perform specific
operations within a data flow graph. Operator processors are essential
building blocks in constructing complex data processing pipelines,
enabling efficient and modular handling of data transformations.
"""

from typing import TYPE_CHECKING

__all__ = [
    "Add",
    "Equals",
    "FloorDiv",
    "GreaterThan",
    "GreaterThanOrEqual",
    "LessThan",
    "LessThanOrEqual",
    "LogicalAnd",
    "LogicalOr",
    "LogicalXOr",
    "Mod",
    "Mul",
    "NotEquals",
    "Pow",
    "Sub",
    "TrueDiv",
    "CollectFeatures",
    "NoOp",
    "BooleanIndexing",
    "SequenceGetItem",
    "SequenceSetItem",
    "SequenceChain",
    "SequenceZip",
    "SequenceContains",
    "SequenceCountOf",
    "SequenceIndexOf",
    "SequenceLength",
    "SequenceMean",
    "SequenceSum",
    "Abs",
    "BooleanInvert",
    "Invert",
    "Neg",
]

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
    from .sequence.access import BooleanIndexing, SequenceGetItem, SequenceSetItem
    from .sequence.multi import SequenceChain, SequenceZip
    from .sequence.query import SequenceContains, SequenceCountOf, SequenceIndexOf
    from .sequence.reduce import SequenceLength, SequenceMean, SequenceSum
    from .unary import Abs, BooleanInvert, Invert, Neg

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        # binary
        "Add": "hyped.nodes._ops.binary",
        "Equals": "hyped.nodes._ops.binary",
        "FloorDiv": "hyped.nodes._ops.binary",
        "GreaterThan": "hyped.nodes._ops.binary",
        "GreaterThanOrEqual": "hyped.nodes._ops.binary",
        "LessThan": "hyped.nodes._ops.binary",
        "LessThanOrEqual": "hyped.nodes._ops.binary",
        "LogicalAnd": "hyped.nodes._ops.binary",
        "LogicalOr": "hyped.nodes._ops.binary",
        "LogicalXOr": "hyped.nodes._ops.binary",
        "Mod": "hyped.nodes._ops.binary",
        "Mul": "hyped.nodes._ops.binary",
        "NotEquals": "hyped.nodes._ops.binary",
        "Pow": "hyped.nodes._ops.binary",
        "Sub": "hyped.nodes._ops.binary",
        "TrueDiv": "hyped.nodes._ops.binary",
        # collect
        "CollectFeatures": "hyped.nodes._ops.collect",
        # noop
        "NoOp": "hyped.nodes._ops.noop",
        # sequence
        "BooleanIndexing": "hyped.nodes._ops.sequence.access",
        "SequenceGetItem": "hyped.nodes._ops.sequence.access",
        "SequenceSetItem": "hyped.nodes._ops.sequence.access",
        "SequenceChain": "hyped.nodes._ops.sequence.multi",
        "SequenceZip": "hyped.nodes._ops.sequence.multi",
        "SequenceContains": "hyped.nodes._ops.sequence.query",
        "SequenceCountOf": "hyped.nodes._ops.sequence.query",
        "SequenceIndexOf": "hyped.nodes._ops.sequence.query",
        "SequenceLength": "hyped.nodes._ops.sequence.reduce",
        "SequenceMean": "hyped.nodes._ops.sequence.reduce",
        "SequenceSum": "hyped.nodes._ops.sequence.reduce",
        # unary
        "Abs": "hyped.nodes._ops.unary",
        "BooleanInvert": "hyped.nodes._ops.unary",
        "Invert": "hyped.nodes._ops.unary",
        "Neg": "hyped.nodes._ops.unary",
        # aggregators
        "SumAggregator": "hyped.nodes._ops.aggregate",
        "MeanAggregator": "hyped.nodes._ops.aggregate",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
    )
