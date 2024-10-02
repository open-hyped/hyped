"""Provides high-level feature operators for data processors.

The operator module defines high-level functions for performing common operations.
These functions delegate the actual processing to specific processor classes and
return references to the resulting features represented by `FeatureRef` instances.

Feature operators are designed to simplify the process of adding processors to a data
flow by providing high-level functions for common feature operations. Each operator
encapsulates the logic for performing specific tasks, such as collecting features from
a collection (e.g., dictionary or list). These functions leverage underlying processor
classes, such as `CollectFeatures`, to execute the desired operations.

Functions:
    - :class:`collect`: Collect features from a given collection.

Usage Example:
    Collect features from a dictionary using the :class:`collect` operator:

    .. code-block:: python

        # Import the collect operator from the module
        from hyped.ops import collect

        # Define the features of the source node
        src_features = datasets.Features({"text": datasets.Value("string")})

        # Initialize a DataFlow instance with the source features
        flow = DataFlow(features=src_features)
        
        # Collect features from the dictionary using the collect operator
        collected_features = collect(
            collection={
                "out": [
                    flow.src_features.text,
                    flow.src_features.text
                ]
            }
        )

        collected_features.out  # work with the collected features

"""

from typing import TYPE_CHECKING

__all__ = [
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "eq",
    "pow_",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and_",
    "or_",
    "xor_",
    "precision_recall_fscore_support",
    "chain",
    "chunk",
    "choice",
    "compress",
    "contains",
    "count_of",
    "get_item",
    "index_of",
    "len_",
    "set_item",
    "zip_",
    "sum_",
    "mean",
    "neg",
    "abs_",
    "invert",
    "collect",
]

if TYPE_CHECKING:
    from .binary import (
        add,
        and_,
        eq,
        floordiv,
        ge,
        gt,
        le,
        lt,
        mod,
        mul,
        ne,
        or_,
        pow_,
        sub,
        truediv,
        xor_,
    )
    from .metrics import precision_recall_fscore_support
    from .sequence import (
        chain,
        choice,
        chunk,
        compress,
        contains,
        count_of,
        get_item,
        index_of,
        len_,
        set_item,
        zip_,
    )
    from .unary import abs_, invert, mean, neg, sum_
    from .utils import collect

else:
    import sys

    from hyped.common.lazy_module import LazyModule

    _lazy_imports = {
        # binary operations
        "add": "hyped.ops.binary",
        "sub": "hyped.ops.binary",
        "mul": "hyped.ops.binary",
        "truediv": "hyped.ops.binary",
        "floordiv": "hyped.ops.binary",
        "mod": "hyped.ops.binary",
        "eq": "hyped.ops.binary",
        "pow_": "hyped.ops.binary",
        "ne": "hyped.ops.binary",
        "lt": "hyped.ops.binary",
        "le": "hyped.ops.binary",
        "gt": "hyped.ops.binary",
        "ge": "hyped.ops.binary",
        "and_": "hyped.ops.binary",
        "or_": "hyped.ops.binary",
        "xor_": "hyped.ops.binary",
        # collect
        "collect": "hyped.ops.utils",
        # metrics
        "precision_recall_fscore_support": "hyped.ops.metrics",
        # sequence operations
        "chain": "hyped.ops.sequence",
        "chunk": "hyped.ops.sequence",
        "choice": "hyped.ops.sequence",
        "compress": "hyped.ops.sequence",
        "contains": "hyped.ops.sequence",
        "count_of": "hyped.ops.sequence",
        "get_item": "hyped.ops.sequence",
        "index_of": "hyped.ops.sequence",
        "len_": "hyped.ops.sequence",
        "set_item": "hyped.ops.sequence",
        "zip_": "hyped.ops.sequence",
        # unary operations
        "sum_": "hyped.ops.unary",
        "mean": "hyped.ops.unary",
        "neg": "hyped.ops.unary",
        "abs_": "hyped.ops.unary",
        "invert": "hyped.ops.unary",
    }

    sys.modules[__name__] = LazyModule(
        __name__,
        __doc__,
        globals()["__file__"],
        __spec__,
        lazy_imports=_lazy_imports,
    )
