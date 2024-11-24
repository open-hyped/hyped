from typing import Generic, TypeVar
from unittest.mock import MagicMock

import pydantic
import pytest

from hyped.core.features.features import (
    BoolFeature,
    Feature,
    PrimitiveFeature,
    SequenceFeature,
    StringFeature,
)
from hyped.core.features.features import _MappingFeature as MappingFeature
from hyped.core.features.features import build_feature_from_annotation, build_feature_from_dtype
from hyped.core.features.reference import FeatureKey, Reference
from hyped.core.features.types import BoolType, MappingType, SequenceType, StringType, Type


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
