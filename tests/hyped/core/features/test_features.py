import inspect
from itertools import chain
from typing import Any, Callable, Generic, TypeVar, get_overloads, get_type_hints
from unittest.mock import MagicMock

import pydantic
import pytest

from hyped.core.features.features import (
    BoolFeature,
    ClassLabelFeature,
    Feature,
    Float32Feature,
    Float64Feature,
    Int8Feature,
    Int16Feature,
    Int32Feature,
    Int64Feature,
    PrimitiveFeature,
    SequenceFeature,
    StringFeature,
    UInt8Feature,
    UInt16Feature,
    UInt32Feature,
    UInt64Feature,
)
from hyped.core.features.features import _MappingFeature as MappingFeature
from hyped.core.features.features import build_feature_from_annotation, build_feature_from_dtype
from hyped.core.features.reference import FeatureKey, Reference
from hyped.core.features.types import (
    UNDEFINED_SEQUENCE_LENGTH,
    BoolType,
    ClassLabelType,
    Float32Type,
    Float64Type,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    MappingType,
    SequenceType,
    StringType,
    Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
)
from hyped.core.graph import DataFlowGraph


def _test_call_to_registered_method(
    feature: Feature,
    fn: Callable,
    registered_fn_name: str,
    args: tuple[Any | Feature],
    expected_return_feature_type: None | type[Feature],
) -> None:
    # create a data flow graph
    graph = DataFlowGraph()
    # add all feature arguments to the source node
    source = graph.add_source_node(
        MappingType.construct(
            {"feature": feature.dtype}
            | {str(i): f.dtype for i, f in enumerate(args) if isinstance(f, Feature)}
        )
    )

    # get the feature and mock the get method function to check execution later
    feature = type(feature)(
        Reference(FeatureKey("feature"), source._node_id, source._graph), feature.dtype
    )
    type(feature).get_method = MagicMock(side_effect=type(feature).get_method)

    # collect all arguments
    args = tuple(
        f
        if not isinstance(f, Feature)
        else type(f)(Reference(FeatureKey(str(i)), source._node_id, source._graph), f.dtype)
        for i, f in enumerate(args)
    )

    out_feature = fn(feature, *args)

    if expected_return_feature_type is not None:
        # make sure the return value matches the expected type
        assert isinstance(out_feature, expected_return_feature_type)
    else:
        # no output expected
        assert out_feature is None

    # make sure the call was forwarded to the right registered method
    type(feature).get_method.assert_called_once_with(registered_fn_name)


class TestPrimitiveFeatures:
    @pytest.mark.parametrize(
        "feature_type, fn, registered_fn_name, args, expected_return_feature_type",
        [
            # Boolean methods
            (BoolFeature, BoolFeature.__invert__, "__invert__", tuple(), BoolFeature),
            (BoolFeature, BoolFeature.__and__, "__and__", (BoolFeature,), BoolFeature),
            (BoolFeature, BoolFeature.__and__, "__and__", (False,), BoolFeature),
            (BoolFeature, BoolFeature.__rand__, "__rand__", (False,), BoolFeature),
            (BoolFeature, BoolFeature.__or__, "__or__", (BoolFeature,), BoolFeature),
            (BoolFeature, BoolFeature.__or__, "__or__", (False,), BoolFeature),
            (BoolFeature, BoolFeature.__ror__, "__ror__", (False,), BoolFeature),
            (BoolFeature, BoolFeature.__xor__, "__xor__", (BoolFeature,), BoolFeature),
            (BoolFeature, BoolFeature.__xor__, "__xor__", (False,), BoolFeature),
            (BoolFeature, BoolFeature.__rxor__, "__rxor__", (False,), BoolFeature),
            # String methods
            (StringFeature, StringFeature.__add__, "__add__", (StringFeature,), StringFeature),
            (StringFeature, StringFeature.__add__, "__add__", ("OTHER_STRING",), StringFeature),
            (StringFeature, StringFeature.__radd__, "__radd__", ("OTHER_STRING",), StringFeature),
            (StringFeature, StringFeature.__mul__, "__mul__", (Int8Feature,), StringFeature),
            (StringFeature, StringFeature.__mul__, "__mul__", (3,), StringFeature),
            (StringFeature, StringFeature.__rmul__, "__rmul__", (3,), StringFeature),
            (StringFeature, StringFeature.__getitem__, "__getitem__", (3,), StringFeature),
            (
                StringFeature,
                StringFeature.__getitem__,
                "__getitem__",
                (slice(1, 3),),
                StringFeature,
            ),
            (StringFeature, StringFeature.__setitem__, "__setitem__", (1, "A"), None),
            (StringFeature, StringFeature.__setitem__, "__setitem__", (slice(1, 3), "AB"), None),
            (StringFeature, StringFeature.upper, "upper", tuple(), StringFeature),
            (StringFeature, StringFeature.lower, "lower", tuple(), StringFeature),
            (StringFeature, StringFeature.capitalize, "capitalize", tuple(), StringFeature),
            (StringFeature, StringFeature.title, "title", tuple(), StringFeature),
            (StringFeature, StringFeature.swapcase, "swapcase", tuple(), StringFeature),
            (StringFeature, StringFeature.startswith, "startswith", ("PATTERN",), BoolFeature),
            (StringFeature, StringFeature.endswith, "endswith", ("PATTERN",), BoolFeature),
            (
                StringFeature,
                StringFeature.replace,
                "replace",
                ("PATTERN", "REPLACEMENT"),
                StringFeature,
            ),
            (StringFeature, StringFeature.find, "find", ("PATTERN",), Int32Feature),
            (StringFeature, StringFeature.split, "split", ("PATTERN",), SequenceFeature),
            (StringFeature, StringFeature.split, "split", ("PATTERN", 3), SequenceFeature),
            (StringFeature, StringFeature.rsplit, "split", ("PATTERN",), SequenceFeature),
            (StringFeature, StringFeature.rsplit, "split", ("PATTERN", 3), SequenceFeature),
            (StringFeature, StringFeature.strip, "strip", tuple(), StringFeature),
            (StringFeature, StringFeature.strip, "strip", ("PATTERN",), StringFeature),
            (StringFeature, StringFeature.rstrip, "rstrip", tuple(), StringFeature),
            (StringFeature, StringFeature.rstrip, "rstrip", ("PATTERN",), StringFeature),
            (StringFeature, StringFeature.lstrip, "lstrip", tuple(), StringFeature),
            (StringFeature, StringFeature.lstrip, "lstrip", ("PATTERN",), StringFeature),
            (StringFeature, StringFeature.format, "format", tuple(), StringFeature),
            # Int8 methods
            (Int8Feature, Int8Feature.__abs__, "__abs__", tuple(), Int8Feature),
            (Int8Feature, Int8Feature.__neg__, "__neg__", tuple(), Int8Feature),
            # addition
            (Int8Feature, Int8Feature.__add__, "__add__", (Int8Feature,), Int8Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (Int16Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (Int32Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (UInt8Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (UInt16Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (UInt32Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (UInt64Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (1.2,), Float64Feature),
            (Int8Feature, Int8Feature.__radd__, "__add__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Int8Feature,), Int8Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Int16Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Int32Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (UInt8Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (UInt16Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (UInt32Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (UInt64Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (Int8Feature, Int8Feature.__rsub__, "__sub__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Int8Feature,), Int8Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Int16Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Int32Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (UInt8Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (UInt16Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (UInt32Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (UInt64Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (Int8Feature, Int8Feature.__rmul__, "__mul__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (Int8Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (Int16Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (Int32Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (Int64Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (UInt8Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (UInt16Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (UInt32Feature,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (UInt64Feature,), Float64Feature),
            (
                Int8Feature,
                Int8Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                Int8Feature,
                Int8Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (Int8Feature, Int8Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (Int8Feature, Int8Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (Int8Feature, Int8Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (Int8Feature,), Int8Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (Int16Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (Int32Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (Int64Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (UInt8Feature,), Int16Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (UInt16Feature,), Int32Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (UInt32Feature,), Int64Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (UInt64Feature,), Int64Feature),
            (
                Int8Feature,
                Int8Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                Int8Feature,
                Int8Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Int8Feature, Int8Feature.__rfloordiv__, "__floordiv__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # Int16 methods
            (Int16Feature, Int16Feature.__abs__, "__abs__", tuple(), Int16Feature),
            (Int16Feature, Int16Feature.__neg__, "__neg__", tuple(), Int16Feature),
            # addition
            (Int16Feature, Int16Feature.__add__, "__add__", (Int8Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (Int16Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (Int32Feature,), Int32Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (UInt8Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (UInt16Feature,), Int32Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (UInt32Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (UInt64Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__add__, "__add__", (1.2,), Float64Feature),
            (Int16Feature, Int16Feature.__radd__, "__add__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (Int16Feature, Int16Feature.__sub__, "__sub__", (Int8Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (Int16Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (Int32Feature,), Int32Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (UInt8Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (UInt16Feature,), Int32Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (UInt32Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (UInt64Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (Int16Feature, Int16Feature.__rsub__, "__sub__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (Int16Feature, Int16Feature.__mul__, "__mul__", (Int8Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (Int16Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (Int32Feature,), Int32Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (UInt8Feature,), Int16Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (UInt16Feature,), Int32Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (UInt32Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (UInt64Feature,), Int64Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (Int16Feature, Int16Feature.__rmul__, "__mul__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (Int16Feature, Int16Feature.__truediv__, "__truediv__", (Int8Feature,), Float64Feature),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (Int16Feature, Int16Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (Int16Feature, Int16Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (Int16Feature, Int16Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (Int16Feature, Int16Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (Int16Feature, Int16Feature.__floordiv__, "__floordiv__", (Int8Feature,), Int16Feature),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int16Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int32Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                Int16Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                Int32Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                Int64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                Int64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                Int16Feature,
                Int16Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (Int16Feature, Int16Feature.__floordiv__, "__floordiv__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Int16Feature, Int16Feature.__rfloordiv__, "__floordiv__", (1,), Int16Feature),
            (Int16Feature, Int16Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # Int32 methods
            (Int32Feature, Int32Feature.__abs__, "__abs__", tuple(), Int32Feature),
            (Int32Feature, Int32Feature.__neg__, "__neg__", tuple(), Int32Feature),
            # addition
            (Int32Feature, Int32Feature.__add__, "__add__", (Int8Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (Int16Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (Int32Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (UInt8Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (UInt16Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (UInt32Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (UInt64Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__add__, "__add__", (1.2,), Float64Feature),
            (Int32Feature, Int32Feature.__radd__, "__add__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (Int32Feature, Int32Feature.__sub__, "__sub__", (Int8Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (Int16Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (Int32Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (UInt8Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (UInt16Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (UInt32Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (UInt64Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (Int32Feature, Int32Feature.__rsub__, "__sub__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (Int32Feature, Int32Feature.__mul__, "__mul__", (Int8Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (Int16Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (Int32Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (UInt8Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (UInt16Feature,), Int32Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (UInt32Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (UInt64Feature,), Int64Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (Int32Feature, Int32Feature.__rmul__, "__mul__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (Int32Feature, Int32Feature.__truediv__, "__truediv__", (Int8Feature,), Float64Feature),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (Int32Feature, Int32Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (Int32Feature, Int32Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (Int32Feature, Int32Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (Int32Feature, Int32Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (Int32Feature, Int32Feature.__floordiv__, "__floordiv__", (Int8Feature,), Int32Feature),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int32Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int32Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                Int32Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                Int32Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                Int64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                Int64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                Int32Feature,
                Int32Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (Int32Feature, Int32Feature.__floordiv__, "__floordiv__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Int32Feature, Int32Feature.__rfloordiv__, "__floordiv__", (1,), Int32Feature),
            (Int32Feature, Int32Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # Int64 methods
            (Int64Feature, Int64Feature.__abs__, "__abs__", tuple(), Int64Feature),
            (Int64Feature, Int64Feature.__neg__, "__neg__", tuple(), Int64Feature),
            # addition
            (Int64Feature, Int64Feature.__add__, "__add__", (Int8Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (Int16Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (Int32Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (UInt8Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (UInt16Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (UInt32Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (UInt64Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__add__, "__add__", (1.2,), Float64Feature),
            (Int64Feature, Int64Feature.__radd__, "__add__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (Int64Feature, Int64Feature.__sub__, "__sub__", (Int8Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (Int16Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (Int32Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (UInt8Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (UInt16Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (UInt32Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (UInt64Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (Int64Feature, Int64Feature.__rsub__, "__sub__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (Int64Feature, Int64Feature.__mul__, "__mul__", (Int8Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (Int16Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (Int32Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (UInt8Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (UInt16Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (UInt32Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (UInt64Feature,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (Int64Feature, Int64Feature.__rmul__, "__mul__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (Int64Feature, Int64Feature.__truediv__, "__truediv__", (Int8Feature,), Float64Feature),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (Int64Feature, Int64Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (Int64Feature, Int64Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (Int64Feature, Int64Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (Int64Feature, Int64Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (Int64Feature, Int64Feature.__floordiv__, "__floordiv__", (Int8Feature,), Int64Feature),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                Int64Feature,
                Int64Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (Int64Feature, Int64Feature.__floordiv__, "__floordiv__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Int64Feature, Int64Feature.__rfloordiv__, "__floordiv__", (1,), Int64Feature),
            (Int64Feature, Int64Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # UInt8 methods
            (UInt8Feature, UInt8Feature.__abs__, "__abs__", tuple(), UInt8Feature),
            (UInt8Feature, UInt8Feature.__neg__, "__neg__", tuple(), Int16Feature),
            # addition
            (UInt8Feature, UInt8Feature.__add__, "__add__", (Int8Feature,), Int16Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (Int16Feature,), Int16Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (Int32Feature,), Int32Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (UInt8Feature,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (UInt16Feature,), UInt16Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (UInt32Feature,), UInt32Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (UInt64Feature,), UInt64Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (-1,), Int32Feature),
            (UInt8Feature, UInt8Feature.__add__, "__add__", (1.2,), Float64Feature),
            (UInt8Feature, UInt8Feature.__radd__, "__add__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (Int8Feature,), Int16Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (Int16Feature,), Int16Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (Int32Feature,), Int32Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (UInt8Feature,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (UInt16Feature,), UInt16Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (UInt32Feature,), UInt32Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (UInt64Feature,), UInt64Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (-1,), Int32Feature),
            (UInt8Feature, UInt8Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (UInt8Feature, UInt8Feature.__rsub__, "__sub__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (Int8Feature,), Int16Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (Int16Feature,), Int16Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (Int32Feature,), Int32Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (UInt8Feature,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (UInt16Feature,), UInt16Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (UInt32Feature,), UInt32Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (UInt64Feature,), UInt64Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (-1,), Int32Feature),
            (UInt8Feature, UInt8Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (UInt8Feature, UInt8Feature.__rmul__, "__mul__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (UInt8Feature, UInt8Feature.__truediv__, "__truediv__", (Int8Feature,), Float64Feature),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (UInt8Feature, UInt8Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (UInt8Feature, UInt8Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (UInt8Feature, UInt8Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (UInt8Feature, UInt8Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (UInt8Feature, UInt8Feature.__floordiv__, "__floordiv__", (Int8Feature,), Int16Feature),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int16Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int32Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                UInt8Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                UInt16Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                UInt32Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                UInt64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                UInt8Feature,
                UInt8Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (UInt8Feature, UInt8Feature.__floordiv__, "__floordiv__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__floordiv__, "__floordiv__", (-1,), Int32Feature),
            (UInt8Feature, UInt8Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (UInt8Feature, UInt8Feature.__rfloordiv__, "__floordiv__", (1,), UInt8Feature),
            (UInt8Feature, UInt8Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # UInt16 methods
            (UInt16Feature, UInt16Feature.__abs__, "__abs__", tuple(), UInt16Feature),
            (UInt16Feature, UInt16Feature.__neg__, "__neg__", tuple(), Int32Feature),
            # addition
            (UInt16Feature, UInt16Feature.__add__, "__add__", (Int8Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (Int16Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (Int32Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (UInt8Feature,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (UInt16Feature,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (UInt32Feature,), UInt32Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (UInt64Feature,), UInt64Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (-1,), Int32Feature),
            (UInt16Feature, UInt16Feature.__add__, "__add__", (1.2,), Float64Feature),
            (UInt16Feature, UInt16Feature.__radd__, "__add__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (Int8Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (Int16Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (Int32Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (UInt8Feature,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (UInt16Feature,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (UInt32Feature,), UInt32Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (UInt64Feature,), UInt64Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (-1,), Int32Feature),
            (UInt16Feature, UInt16Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (UInt16Feature, UInt16Feature.__rsub__, "__sub__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (Int8Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (Int16Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (Int32Feature,), Int32Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (UInt8Feature,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (UInt16Feature,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (UInt32Feature,), UInt32Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (UInt64Feature,), UInt64Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (-1,), Int32Feature),
            (UInt16Feature, UInt16Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (UInt16Feature, UInt16Feature.__rmul__, "__mul__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (Int8Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (UInt16Feature, UInt16Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (UInt16Feature, UInt16Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (UInt16Feature, UInt16Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (UInt16Feature, UInt16Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (Int8Feature,),
                Int32Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int32Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int32Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                UInt16Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                UInt16Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                UInt32Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                UInt64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                UInt16Feature,
                UInt16Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (UInt16Feature, UInt16Feature.__floordiv__, "__floordiv__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__floordiv__, "__floordiv__", (-1,), Int32Feature),
            (UInt16Feature, UInt16Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (UInt16Feature, UInt16Feature.__rfloordiv__, "__floordiv__", (1,), UInt16Feature),
            (UInt16Feature, UInt16Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # UInt32 methods
            (UInt32Feature, UInt32Feature.__abs__, "__abs__", tuple(), UInt32Feature),
            (UInt32Feature, UInt32Feature.__neg__, "__neg__", tuple(), Int64Feature),
            # addition
            (UInt32Feature, UInt32Feature.__add__, "__add__", (Int8Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (Int16Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (Int32Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (UInt8Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (UInt16Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (UInt32Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (UInt64Feature,), UInt64Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (-1,), Int64Feature),
            (UInt32Feature, UInt32Feature.__add__, "__add__", (1.2,), Float64Feature),
            (UInt32Feature, UInt32Feature.__radd__, "__add__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (Int8Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (Int16Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (Int32Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (UInt8Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (UInt16Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (UInt32Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (UInt64Feature,), UInt64Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (-1,), Int64Feature),
            (UInt32Feature, UInt32Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (UInt32Feature, UInt32Feature.__rsub__, "__sub__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (Int8Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (Int16Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (Int32Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (UInt8Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (UInt16Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (UInt32Feature,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (UInt64Feature,), UInt64Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (-1,), Int64Feature),
            (UInt32Feature, UInt32Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (UInt32Feature, UInt32Feature.__rmul__, "__mul__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (Int8Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (UInt32Feature, UInt32Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (UInt32Feature, UInt32Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (UInt32Feature, UInt32Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (UInt32Feature, UInt32Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (Int8Feature,),
                Int64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                UInt32Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                UInt32Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                UInt32Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                UInt64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                UInt32Feature,
                UInt32Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (UInt32Feature, UInt32Feature.__floordiv__, "__floordiv__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__floordiv__, "__floordiv__", (-1,), Int64Feature),
            (UInt32Feature, UInt32Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (UInt32Feature, UInt32Feature.__rfloordiv__, "__floordiv__", (1,), UInt32Feature),
            (UInt32Feature, UInt32Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # UInt64 methods
            (UInt64Feature, UInt64Feature.__abs__, "__abs__", tuple(), UInt64Feature),
            (UInt64Feature, UInt64Feature.__neg__, "__neg__", tuple(), Int64Feature),
            # addition
            (UInt64Feature, UInt64Feature.__add__, "__add__", (Int8Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (Int16Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (Int32Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (UInt8Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (UInt16Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (UInt32Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (UInt64Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (-1,), Int64Feature),
            (UInt64Feature, UInt64Feature.__add__, "__add__", (1.2,), Float64Feature),
            (UInt64Feature, UInt64Feature.__radd__, "__add__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (Int8Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (Int16Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (Int32Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (UInt8Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (UInt16Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (UInt32Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (UInt64Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (-1,), Int64Feature),
            (UInt64Feature, UInt64Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (UInt64Feature, UInt64Feature.__rsub__, "__sub__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (Int8Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (Int16Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (Int32Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (UInt8Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (UInt16Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (UInt32Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (UInt64Feature,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (-1,), Int64Feature),
            (UInt64Feature, UInt64Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (UInt64Feature, UInt64Feature.__rmul__, "__mul__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (Int8Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (UInt64Feature, UInt64Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (UInt64Feature, UInt64Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (UInt64Feature, UInt64Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (UInt64Feature, UInt64Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (Int8Feature,),
                Int64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                UInt64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                UInt64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                UInt64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                UInt64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                UInt64Feature,
                UInt64Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (UInt64Feature, UInt64Feature.__floordiv__, "__floordiv__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__floordiv__, "__floordiv__", (-1,), Int64Feature),
            (UInt64Feature, UInt64Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (UInt64Feature, UInt64Feature.__rfloordiv__, "__floordiv__", (1,), UInt64Feature),
            (UInt64Feature, UInt64Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # float32 methods
            (Float32Feature, Float32Feature.__abs__, "__abs__", tuple(), Float32Feature),
            (Float32Feature, Float32Feature.__neg__, "__neg__", tuple(), Float32Feature),
            # addition
            (Float32Feature, Float32Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (Int8Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (Int16Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (Int32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (Int64Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (UInt8Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (UInt16Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (UInt32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (UInt64Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__add__, "__add__", (1.2,), Float32Feature),
            (Float32Feature, Float32Feature.__radd__, "__add__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__radd__, "__add__", (1.2,), Float32Feature),
            # subtraction
            (Float32Feature, Float32Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (Int8Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (Int16Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (Int32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (Int64Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (UInt8Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (UInt16Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (UInt32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (UInt64Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__sub__, "__sub__", (1.2,), Float32Feature),
            (Float32Feature, Float32Feature.__rsub__, "__sub__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__rsub__, "__sub__", (1.2,), Float32Feature),
            # multiplication
            (Float32Feature, Float32Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (Int8Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (Int16Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (Int32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (Int64Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (UInt8Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (UInt16Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (UInt32Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (UInt64Feature,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__mul__, "__mul__", (1.2,), Float32Feature),
            (Float32Feature, Float32Feature.__rmul__, "__mul__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__rmul__, "__mul__", (1.2,), Float32Feature),
            # true division
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (Int8Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float32Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float32Feature,
            ),
            (Float32Feature, Float32Feature.__truediv__, "__truediv__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__truediv__, "__truediv__", (1.2,), Float32Feature),
            (Float32Feature, Float32Feature.__rtruediv__, "__truediv__", (1,), Float32Feature),
            (Float32Feature, Float32Feature.__rtruediv__, "__truediv__", (1.2,), Float32Feature),
            # floor division
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (Int8Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                Int64Feature,
            ),
            (
                Float32Feature,
                Float32Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                Int64Feature,
            ),
            (Float32Feature, Float32Feature.__floordiv__, "__floordiv__", (1,), Int64Feature),
            (Float32Feature, Float32Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Float32Feature, Float32Feature.__floordiv__, "__floordiv__", (1,), Int64Feature),
            (Float32Feature, Float32Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Float32Feature, Float32Feature.__rfloordiv__, "__floordiv__", (1,), Int64Feature),
            (Float32Feature, Float32Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
            # float64 methods
            (Float64Feature, Float64Feature.__abs__, "__abs__", tuple(), Float64Feature),
            (Float64Feature, Float64Feature.__neg__, "__neg__", tuple(), Float64Feature),
            # addition
            (Float64Feature, Float64Feature.__add__, "__add__", (Float32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (Float32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (Int8Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (Int16Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (Int32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (Int64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (UInt8Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (UInt16Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (UInt32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (UInt64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__add__, "__add__", (1.2,), Float64Feature),
            (Float64Feature, Float64Feature.__radd__, "__add__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (Float64Feature, Float64Feature.__sub__, "__sub__", (Float32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (Float32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (Int8Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (Int16Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (Int32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (Int64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (UInt8Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (UInt16Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (UInt32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (UInt64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__sub__, "__sub__", (1.2,), Float64Feature),
            (Float64Feature, Float64Feature.__rsub__, "__sub__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (Float64Feature, Float64Feature.__mul__, "__mul__", (Float32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (Float32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (Int8Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (Int16Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (Int32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (Int64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (UInt8Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (UInt16Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (UInt32Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (UInt64Feature,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__mul__, "__mul__", (1.2,), Float64Feature),
            (Float64Feature, Float64Feature.__rmul__, "__mul__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (Int8Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (Float64Feature, Float64Feature.__truediv__, "__truediv__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (Float64Feature, Float64Feature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (Float64Feature, Float64Feature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (Int8Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                Int64Feature,
            ),
            (
                Float64Feature,
                Float64Feature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                Int64Feature,
            ),
            (Float64Feature, Float64Feature.__floordiv__, "__floordiv__", (1,), Int64Feature),
            (Float64Feature, Float64Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Float64Feature, Float64Feature.__floordiv__, "__floordiv__", (1,), Int64Feature),
            (Float64Feature, Float64Feature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (Float64Feature, Float64Feature.__rfloordiv__, "__floordiv__", (1,), Int64Feature),
            (Float64Feature, Float64Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
        ],
    )
    def test_primitive_feature_method(
        self,
        feature_type: type[PrimitiveFeature],
        fn: Callable,
        registered_fn_name: str,
        args: tuple[Any | type[PrimitiveFeature]],
        expected_return_feature_type: type[PrimitiveFeature],
    ) -> None:
        # create mock instances of the feature types
        feature = feature_type(Reference(), feature_type._expected_dtype)
        args = tuple(
            val(Reference(), val._expected_dtype)
            if isinstance(val, type) and issubclass(val, PrimitiveFeature)
            else val
            for val in args
        )

        _test_call_to_registered_method(
            feature=feature,
            fn=fn,
            registered_fn_name=registered_fn_name,
            args=args,
            expected_return_feature_type=expected_return_feature_type,
        )

        # find a candidate function that matches the inputs
        for candidate_fn in chain(get_overloads(fn), [fn]):
            # get the parameter order of the arguments of the candidate function
            # and remove the self argument
            parameter_order = list(inspect.signature(candidate_fn).parameters.keys())
            parameter_order = parameter_order[1:]
            # get type hints of candidate function
            annotations = get_type_hints(candidate_fn)
            return_annotation = annotations.pop("return")

            # check if all arguments match the signature
            for param, arg in zip(parameter_order, args):
                try:
                    if not ((annotations[param] is Any) or isinstance(arg, annotations[param])):
                        break
                except TypeError:
                    break

            else:
                try:
                    # all params match the candidate function signature
                    assert issubclass(
                        return_annotation, expected_return_feature_type
                    ), f"{return_annotation} != {expected_return_feature_type}"
                except TypeError:
                    pass

                break

    @pytest.mark.parametrize(
        "fn, registered_fn_name, args, expected_return_feature_type",
        [
            # ClassLabel
            (ClassLabelFeature.__abs__, "__abs__", tuple(), Int64Feature),
            (ClassLabelFeature.__neg__, "__neg__", tuple(), Int64Feature),
            # addition
            (ClassLabelFeature.__add__, "__add__", (ClassLabelFeature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (Int8Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (Int16Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (Int32Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (Int64Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (UInt8Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (UInt16Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (UInt32Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (UInt64Feature,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (ClassLabelFeature.__add__, "__add__", (Float64Feature,), Float64Feature),
            (ClassLabelFeature.__add__, "__add__", (1,), Int64Feature),
            (ClassLabelFeature.__add__, "__add__", (1.2,), Float64Feature),
            (ClassLabelFeature.__radd__, "__add__", (1,), Int64Feature),
            (ClassLabelFeature.__radd__, "__add__", (1.2,), Float64Feature),
            # subtraction
            (ClassLabelFeature.__sub__, "__sub__", (ClassLabelFeature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (Int8Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (Int16Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (Int32Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (Int64Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (UInt8Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (UInt16Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (UInt32Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (UInt64Feature,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (ClassLabelFeature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (1,), Int64Feature),
            (ClassLabelFeature.__sub__, "__sub__", (1.2,), Float64Feature),
            (ClassLabelFeature.__rsub__, "__sub__", (1,), Int64Feature),
            (ClassLabelFeature.__rsub__, "__sub__", (1.2,), Float64Feature),
            # multiplication
            (ClassLabelFeature.__mul__, "__mul__", (ClassLabelFeature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (Int8Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (Int16Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (Int32Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (Int64Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (UInt8Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (UInt16Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (UInt32Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (UInt64Feature,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (ClassLabelFeature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (1,), Int64Feature),
            (ClassLabelFeature.__mul__, "__mul__", (1.2,), Float64Feature),
            (ClassLabelFeature.__rmul__, "__mul__", (1,), Int64Feature),
            (ClassLabelFeature.__rmul__, "__mul__", (1.2,), Float64Feature),
            # true division
            (ClassLabelFeature.__truediv__, "__truediv__", (ClassLabelFeature,), Float64Feature),
            (ClassLabelFeature.__truediv__, "__truediv__", (Int8Feature,), Float64Feature),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (Int16Feature,),
                Float64Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (Int32Feature,),
                Float64Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (Int64Feature,),
                Float64Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (UInt8Feature,),
                Float64Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (UInt16Feature,),
                Float64Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (UInt32Feature,),
                Float64Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (UInt64Feature,),
                Float64Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (Float32Feature,),
                Float32Feature,
            ),
            (
                ClassLabelFeature.__truediv__,
                "__truediv__",
                (Float64Feature,),
                Float64Feature,
            ),
            (ClassLabelFeature.__truediv__, "__truediv__", (1,), Float64Feature),
            (ClassLabelFeature.__truediv__, "__truediv__", (1.2,), Float64Feature),
            (ClassLabelFeature.__rtruediv__, "__truediv__", (1,), Float64Feature),
            (ClassLabelFeature.__rtruediv__, "__truediv__", (1.2,), Float64Feature),
            # floor division
            (ClassLabelFeature.__floordiv__, "__floordiv__", (ClassLabelFeature,), Int64Feature),
            (ClassLabelFeature.__floordiv__, "__floordiv__", (Int8Feature,), Int64Feature),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (Int16Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (Int32Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (Int64Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (UInt8Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (UInt16Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (UInt32Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (UInt64Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (Float32Feature,),
                Int64Feature,
            ),
            (
                ClassLabelFeature.__floordiv__,
                "__floordiv__",
                (Float64Feature,),
                Int64Feature,
            ),
            (ClassLabelFeature.__floordiv__, "__floordiv__", (1,), Int64Feature),
            (ClassLabelFeature.__floordiv__, "__floordiv__", (1.2,), Int64Feature),
            (ClassLabelFeature.__rfloordiv__, "__floordiv__", (1,), Int64Feature),
            (ClassLabelFeature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
        ],
    )
    def test_class_label_feature_method(
        self,
        fn: Callable,
        registered_fn_name: str,
        args: tuple[Any | type[PrimitiveFeature]],
        expected_return_feature_type: type[PrimitiveFeature],
    ) -> None:
        dtype = ClassLabelType(names=("A", "B"))
        # create mock instances of the feature types
        feature = ClassLabelFeature(Reference(), dtype)
        args = tuple(
            ClassLabelFeature(Reference(), dtype)
            if isinstance(val, type) and issubclass(val, ClassLabelFeature)
            else val(Reference(), val._expected_dtype)
            if isinstance(val, type) and issubclass(val, PrimitiveFeature)
            else val
            for val in args
        )

        _test_call_to_registered_method(
            feature=feature,
            fn=fn,
            registered_fn_name=registered_fn_name,
            args=args,
            expected_return_feature_type=expected_return_feature_type,
        )


class TestClassLabelFeature:
    def test_post_init(self) -> None:
        ref = Reference()
        dtype = ClassLabelType(names=("labelA", "labelB"))

        # base class label feature allows arbitrary label names
        ClassLabelFeature(ref, dtype=dtype)

        class CustomClassLabel(ClassLabelFeature):
            labelA = 0
            labelB = 1

        # labels in dtype match expectation
        CustomClassLabel(ref, dtype=dtype)

        class CustomClassLabel(ClassLabelFeature):
            labelA = 0

        # labels in dtype are contained in dtype
        CustomClassLabel(ref, dtype=dtype)

        class CustomClassLabel(ClassLabelFeature):
            labelB = 1

        # labels in dtype are contained in dtype
        CustomClassLabel(ref, dtype=dtype)

        class CustomClassLabel(ClassLabelFeature):
            labelA = 0
            labelB = 1
            invalid_label = 2

        # custom class label contains invalid label not contained in dtype
        with pytest.raises(RuntimeError):
            CustomClassLabel(ref, dtype=dtype)

        class CustomClassLabel(ClassLabelFeature):
            invalid_label = 0
            labelB = 1

        # custom class label contains invalid label not contained in dtype
        with pytest.raises(RuntimeError):
            CustomClassLabel(ref, dtype=dtype)

    def test_from_names(self) -> None:
        ref = Reference()
        dtype = ClassLabelType(names=("labelA", "labelB"))

        CustomClassLabel = ClassLabelFeature.from_names(["labelA", "labelB"])
        CustomClassLabel(ref, dtype)

        inst = pydantic.TypeAdapter(CustomClassLabel).validate_python(ref)
        assert isinstance(inst, CustomClassLabel)
        assert inst.dtype.names == dtype.names

        CustomClassLabel = ClassLabelFeature.from_names(["invalid_label"])
        with pytest.raises(RuntimeError):
            CustomClassLabel(ref, dtype)

    def test_build_class_label_dtype(self) -> None:
        dtype = ClassLabelFeature._build_class_label_dtype()
        assert len(dtype) == 0

        class CustomClassLabel(ClassLabelFeature):
            labelA = 0
            labelB = 1

        dtype = CustomClassLabel._build_class_label_dtype()
        assert dtype.names == ("labelA", "labelB")

        class CustomClassLabel(ClassLabelFeature):
            labelA = 1
            labelB = 0

        dtype = CustomClassLabel._build_class_label_dtype()
        assert dtype.names == ("labelB", "labelA")

        class CustomClassLabel(ClassLabelFeature):
            labelA = 0
            labelB = 2

        with pytest.warns(UserWarning):
            dtype = CustomClassLabel._build_class_label_dtype()
            assert dtype.names == ("labelA", "UNDEF", "labelB")

    def test_pydantic_core_schema(self) -> None:
        ref = Reference()
        dtype = ClassLabelType(names=("labelA", "labelB"))
        inst = ClassLabelFeature(ref, dtype)

        class CustomClassLabel(ClassLabelFeature):
            labelA = 0
            labelB = 1

        # base type supports arbitrary label names
        adapter = pydantic.TypeAdapter(ClassLabelFeature)
        assert adapter.validate_python(inst) == inst

        # cast to custom type
        adapter = pydantic.TypeAdapter(CustomClassLabel)
        assert isinstance(adapter.validate_python(inst), CustomClassLabel)

        # cast to custom type
        adapter = pydantic.TypeAdapter(CustomClassLabel)
        inferred_dtype = adapter.validate_python(ref).dtype
        assert inferred_dtype.names == dtype.names

        with pytest.raises(pydantic.ValidationError):
            # unable to infer labels
            pydantic.TypeAdapter(ClassLabelFeature).validate_python(ref)


class TestSequenceFeature:
    def test_subclassing(self) -> None:
        with pytest.raises(EnvironmentError):

            class MySequence(SequenceFeature):
                ...

    def test_length(self) -> None:
        # create a mock sequence with fixed length
        dtype = MagicMock(spec=SequenceType, __len__=lambda _: 5)
        sequence = SequenceFeature[BoolFeature](MagicMock(), dtype)
        # check return value of length method is length as integer
        assert isinstance(sequence.length(), int) and (sequence.length() == 5)

        # create a mock sequence with undefined length
        dtype = MagicMock(spec=SequenceType, __len__=lambda _: UNDEFINED_SEQUENCE_LENGTH)
        sequence = SequenceFeature[BoolFeature](MagicMock(), dtype)

        sequence.execute_method = MagicMock()
        # check return value of length method is length as integer
        assert sequence.length() == sequence.execute_method.return_value
        sequence.execute_method.assert_called_once_with("length")

        # use .length() instead
        with pytest.raises(EnvironmentError):
            len(sequence)

    @pytest.mark.parametrize(
        "fn, registered_fn_name",
        [
            (SequenceFeature.min, "min"),
            (SequenceFeature.max, "max"),
            (SequenceFeature.sum, "sum"),
        ],
    )
    def test_methods(self, fn: Callable, registered_fn_name: str) -> None:
        # create a mock sequence with fixed length
        dtype = MagicMock(spec=SequenceType)
        sequence = SequenceFeature[BoolFeature](MagicMock(), dtype)
        # mock execute method function
        sequence.execute_method = MagicMock()
        # call function
        fn(sequence)
        # check registered function call
        sequence.execute_method.assert_called_once_with(registered_fn_name)

    def test_get_item(self) -> None:
        ref = Reference()
        dtype = MagicMock(spec=SequenceType, __getitem__=MagicMock(return_value=BoolType))
        sequence = SequenceFeature[BoolFeature](ref, dtype)

        for i in range(10):
            # reset mock and index sequence
            dtype.__getitem__.reset_mock()
            item = sequence[i]
            # check item
            assert isinstance(item, PrimitiveFeature) and item.dtype == BoolType
            assert item.ref == Reference(FeatureKey(i), ref._node_id, ref._graph)
            dtype.__getitem__.assert_called_once_with(i)

    def test_get_slice(self) -> None:
        ref = Reference()
        dtype = MagicMock(spec=SequenceType)
        dtype.__getitem__ = MagicMock(return_value=dtype)
        # create mock sequence
        sequence = SequenceFeature[BoolFeature](ref, dtype)

        # reset mock and slice sequence
        dtype.__getitem__.reset_mock()
        subseq = sequence[:]
        # check sliced sequence
        assert isinstance(subseq, SequenceFeature) and (subseq.dtype == dtype)
        assert subseq.ref == Reference(FeatureKey(slice(None)), ref._node_id, ref._graph)
        dtype.__getitem__.assert_called_once_with(slice(None))

        # reset mock and slice sequence
        dtype.__getitem__.reset_mock()
        subseq = sequence[3:9:2]
        # check sliced sequence
        assert isinstance(subseq, SequenceFeature) and (subseq.dtype == dtype)
        assert subseq.ref == Reference(FeatureKey(slice(3, 9, 2)), ref._node_id, ref._graph)
        dtype.__getitem__.assert_called_once_with(slice(3, 9, 2))

    def test_pydantic_core_schema(self) -> None:
        # test validation of sequence feature
        seq = SequenceFeature(Reference(), SequenceType(BoolType))
        adapter = pydantic.TypeAdapter(SequenceFeature)
        assert seq == adapter.validate_python(seq)

        # test validation of strongly-typed sequence feature
        seq = SequenceFeature(Reference(), SequenceType(BoolType))
        adapter = pydantic.TypeAdapter(SequenceFeature[BoolFeature])
        assert seq == adapter.validate_python(seq)

        # test validation error on value type mismatch
        seq = SequenceFeature(Reference(), SequenceType(BoolType))
        with pytest.raises(pydantic.ValidationError):
            adapter = pydantic.TypeAdapter(SequenceFeature[StringType])
            adapter.validate_python(seq)

        # test create sequence feature from reference
        seq = SequenceFeature(Reference(), SequenceType(BoolType))
        adapter = pydantic.TypeAdapter(SequenceFeature[BoolFeature])
        assert adapter.validate_python(Reference()) == seq

        # cannot infer sequence feature type from annotation
        with pytest.raises(RuntimeError):
            adapter = pydantic.TypeAdapter(SequenceFeature)
            adapter.validate_python(Reference())


class TestMappingFeature:
    def test_post_init(self) -> None:
        ref = Reference()
        dtype = MappingType.construct({"fieldA": BoolType, "fieldB": StringType})

        # base mapping feature allows arbitrary fields
        MappingFeature(ref, dtype=dtype)

        class CustomMappingFeature(MappingFeature):
            fieldA: BoolFeature
            fieldB: StringFeature

        # fields in dtype match expectation
        CustomMappingFeature(ref, dtype)

        class CustomMappingFeature(MappingFeature):
            invalid_fieldA: BoolFeature
            fieldB: StringFeature

        with pytest.raises(KeyError):
            # fields in dtype don't match expectation
            CustomMappingFeature(ref, dtype)

        class CustomMappingFeature(MappingFeature):
            fieldA: BoolFeature
            fieldB: StringFeature
            fieldC: StringFeature

        with pytest.raises(KeyError):
            # fields in dtype don't match expectation
            CustomMappingFeature(ref, dtype)

    def test_get_item(self) -> None:
        ref = Reference()
        dtype = MappingType.construct({"fieldA": BoolType, "fieldB": StringType})
        # base mapping feature allows arbitrary fields
        mapping = MappingFeature(ref, dtype=dtype)

        ref = Reference()
        dtype = MagicMock(spec=MappingType, __getitem__=MagicMock(return_value=BoolType))
        mapping = MappingFeature(ref, dtype)

        for key in ["fieldA", "fieldB"]:
            # reset mock and index sequence
            dtype.__getitem__.reset_mock()
            item = mapping[key]
            # check item
            assert isinstance(item, PrimitiveFeature) and item.dtype == BoolType
            assert item.ref == Reference(FeatureKey(key), ref._node_id, ref._graph)
            dtype.__getitem__.assert_called_once_with(key)

    def test_pydantic_core_schema(self) -> None:
        # create mapping feature instance
        ref = Reference()
        dtype = MappingType.construct({"fieldA": BoolType, "fieldB": StringType})
        mapping = MappingFeature(ref, dtype=dtype)

        class CustomMappingFeature(MappingFeature):
            fieldA: BoolFeature
            fieldB: StringFeature

        T = TypeVar("T")

        class GenericMappingFeature(MappingFeature, Generic[T]):
            fieldA: BoolFeature
            fieldB: T

        class InvalidCustomMappingFeature(MappingFeature):
            fieldA: StringFeature
            fieldB: StringFeature

        # base type supports arbitrary structures
        adapter = pydantic.TypeAdapter(MappingFeature)
        assert adapter.validate_python(mapping) == mapping

        # validate mapping matches fields
        mapping = MappingFeature(ref, dtype=dtype)
        adapter = pydantic.TypeAdapter(CustomMappingFeature)
        assert adapter.validate_python(mapping) == mapping

        # validate and convert to specific mapping type
        mapping = MappingFeature(ref, dtype=dtype)
        adapter = pydantic.TypeAdapter(CustomMappingFeature)
        validated_mapping = adapter.validate_python(mapping, context={"strict": True})
        assert isinstance(validated_mapping, CustomMappingFeature)

        # create mapping feature instance from reference
        adapter = pydantic.TypeAdapter(CustomMappingFeature)
        validated_mapping = adapter.validate_python(ref)
        assert validated_mapping.ref == ref
        assert validated_mapping.dtype == dtype

        # validate generic mapping type
        mapping = MappingFeature(ref, dtype=dtype)
        adapter = pydantic.TypeAdapter(GenericMappingFeature[StringFeature])
        assert adapter.validate_python(mapping) == mapping

        # cannot infer mapping fields
        adapter = pydantic.TypeAdapter(MappingFeature)
        with pytest.raises(RuntimeError):
            adapter.validate_python(ref)

        adapter = pydantic.TypeAdapter(InvalidCustomMappingFeature)
        with pytest.raises(pydantic.ValidationError):
            adapter.validate_python(mapping)

        # validate invalid generic mapping type
        adapter = pydantic.TypeAdapter(GenericMappingFeature[BoolFeature])
        with pytest.raises(pydantic.ValidationError):
            adapter.validate_python(mapping)


@pytest.mark.parametrize(
    "dtype,expected_feature_type",
    [
        (BoolType, PrimitiveFeature),
        (StringType, PrimitiveFeature),
        (UInt8Type, PrimitiveFeature),
        (UInt16Type, PrimitiveFeature),
        (UInt32Type, PrimitiveFeature),
        (UInt64Type, PrimitiveFeature),
        (Int8Type, PrimitiveFeature),
        (Int16Type, PrimitiveFeature),
        (Int32Type, PrimitiveFeature),
        (Int64Type, PrimitiveFeature),
        (Float32Type, PrimitiveFeature),
        (Float64Type, PrimitiveFeature),
        (ClassLabelType(names=("A", "B")), ClassLabelFeature),
        (SequenceType(BoolType), SequenceFeature),
        (MappingType(tuple()), MappingFeature),
    ],
)
def test_build_feature_from_dtype(dtype: Type, expected_feature_type: type[Feature]) -> None:
    assert isinstance(build_feature_from_dtype(Reference(), dtype), expected_feature_type)
    # type error on invalid data type
    with pytest.raises(TypeError):
        build_feature_from_dtype(Reference(), object())


@pytest.mark.parametrize(
    "annotation, expected_dtype",
    [
        # trivial cases
        (BoolFeature, BoolType),
        (StringFeature, StringType),
        (Int8Feature, Int8Type),
        (Int16Feature, Int16Type),
        (Int32Feature, Int32Type),
        (Int64Feature, Int64Type),
        (UInt8Feature, UInt8Type),
        (UInt16Feature, UInt16Type),
        (UInt32Feature, UInt32Type),
        (UInt64Feature, UInt64Type),
        (Float32Feature, Float32Type),
        (Float64Feature, Float64Type),
        (type("CustomLabel", (ClassLabelFeature,), {"A": 0}), ClassLabelType(names=("A",))),
        # union cases, always selects first fitting type
        (BoolFeature | Int8Feature, BoolType),
        (Int8Feature | BoolFeature, Int8Type),
        (int | Int32Feature, Int32Type),
    ],
)
def test_build_feature_from_annotation(annotation: Any, expected_dtype: Type) -> None:
    feature = build_feature_from_annotation(Reference(), annotation)
    assert isinstance(feature, Feature)
    assert feature.dtype == expected_dtype


def test_build_feature_from_generic_annotation() -> None:
    T = TypeVar("T")
    # annotation is only typevar
    feature = build_feature_from_annotation(Reference(), T, typevar_mapping={T: BoolType})
    assert isinstance(feature, PrimitiveFeature)
    assert feature.dtype == BoolType
    # generic annotation
    feature = build_feature_from_annotation(
        Reference(), SequenceFeature[T], typevar_mapping={T: BoolType}
    )
    assert isinstance(feature, SequenceFeature)
    assert feature.dtype.value_type == BoolType
