import inspect
from itertools import chain
from typing import Any, Callable, Generic, TypeVar, get_overloads, get_type_hints
from unittest.mock import MagicMock

import pydantic
import pytest

from hyped.core.features.features import (
    BoolFeature,
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
from hyped.core.features.types import BoolType, MappingType, SequenceType, StringType, Type
from hyped.core.graph import DataFlowGraph


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
            # TODO: output type is not clear
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
        ],
    )
    def test_primitive_feature_method(
        self,
        feature_type: type[PrimitiveFeature],
        fn: Callable,
        registered_fn_name: str,
        args: tuple[Any | type[PrimitiveFeature]],
        expected_return_feature_type: None | type[PrimitiveFeature],
    ) -> None:
        # create a data flow graph
        graph = DataFlowGraph()
        # add all feature arguments to the source node
        source = graph.add_source_node(
            MappingType.from_dict(
                {"feature": feature_type._expected_dtype}
                | {
                    str(i): ftype._expected_dtype
                    for i, ftype in enumerate(args)
                    if isinstance(ftype, type) and issubclass(ftype, PrimitiveFeature)
                }
            )
        )

        # get the feature and mock the get method function to check execution later
        feature = feature_type(
            Reference(FeatureKey("feature"), source._node_id, source._graph),
            feature_type._expected_dtype,
        )
        feature.get_method = MagicMock(side_effect=feature.get_method)

        # collect all arguments
        args = tuple(
            ftype
            if not (isinstance(ftype, type) and issubclass(ftype, PrimitiveFeature))
            else ftype(
                Reference(FeatureKey(str(i)), source._node_id, source._graph), ftype._expected_dtype
            )
            for i, ftype in enumerate(args)
        )

        out_feature = fn(feature, *args)

        if expected_return_feature_type is not None:
            # make sure the return value matches the expected type
            assert isinstance(out_feature, expected_return_feature_type)
        else:
            # no output expected
            assert out_feature is None

        # make sure the call was forwarded to the right registered method
        feature.get_method.assert_called_once_with(registered_fn_name)

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


class TestSequenceFeature:
    def test_subclassing(self) -> None:
        with pytest.raises(EnvironmentError):

            class MySequence(SequenceFeature):
                ...

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
        dtype = MappingType.from_dict({"fieldA": BoolType, "fieldB": StringType})

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
        dtype = MappingType.from_dict({"fieldA": BoolType, "fieldB": StringType})
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
        dtype = MappingType.from_dict({"fieldA": BoolType, "fieldB": StringType})
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
        (SequenceType(BoolType), SequenceFeature),
        (MappingType(tuple()), MappingFeature),
    ],
)
def test_build_feature_from_dtype(dtype: Type, expected_feature_type: type[Feature]) -> None:
    assert isinstance(build_feature_from_dtype(Reference(), dtype), expected_feature_type)
    # type error on invalid data type
    with pytest.raises(TypeError):
        build_feature_from_dtype(Reference(), object())


def test_build_feature_from_annotation() -> None:
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
