import inspect
from typing import Any, Callable, Generic, TypeVar
from unittest.mock import MagicMock

import pydantic
import pytest

from hyped.core.features.features import (
    BoolFeature,
    Feature,
    Float16Feature,
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
            (Int8Feature, Int8Feature.__add__, "__add__", (Float16Feature,), Float16Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (Float32Feature,), Float32Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (Float64Feature,), Float64Feature),
            # TODO: output type is not clear
            (Int8Feature, Int8Feature.__add__, "__add__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__add__, "__add__", (1.2,), Int8Feature),
            (Int8Feature, Int8Feature.__radd__, "__add__", (1,), Int32Feature),
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
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Float16Feature,), Float16Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Float32Feature,), Float32Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (Float64Feature,), Float64Feature),
            # TODO: output type is not clear
            (Int8Feature, Int8Feature.__sub__, "__sub__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__sub__, "__sub__", (1.2,), Int8Feature),
            (Int8Feature, Int8Feature.__rsub__, "__sub__", (1,), Int32Feature),
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
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Float16Feature,), Float16Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Float32Feature,), Float32Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (Float64Feature,), Float64Feature),
            # TODO: output type is not clear
            (Int8Feature, Int8Feature.__mul__, "__mul__", (1,), Int8Feature),
            (Int8Feature, Int8Feature.__mul__, "__mul__", (1.2,), Int8Feature),
            (Int8Feature, Int8Feature.__rmul__, "__mul__", (1,), Int32Feature),
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
                (Float16Feature,),
                Float16Feature,
            ),
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
                (Float16Feature,),
                Int64Feature,
            ),
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
            (Int8Feature, Int8Feature.__floordiv__, "__floordiv__", (1.2,), Int8Feature),
            # TODO: output type is not clear
            (Int8Feature, Int8Feature.__rfloordiv__, "__floordiv__", (1,), Int32Feature),
            (Int8Feature, Int8Feature.__rfloordiv__, "__floordiv__", (1.2,), Int64Feature),
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
        args = (
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
