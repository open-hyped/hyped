from typing import Callable

import pyarrow as pa
import pytest

from hyped.core.features.types import (
    UNDEFINED_SEQUENCE_LENGTH,
    BoolType,
    Float16Type,
    Float32Type,
    Float64Type,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    MappingType,
    PrimitiveType,
    SequenceType,
    StringType,
    Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    build_type_from_dict,
)


@pytest.mark.parametrize(
    "dtype,arrow_type_checker",
    [
        (BoolType, pa.types.is_boolean),
        (StringType, pa.types.is_string),
        (Int8Type, pa.types.is_int8),
        (Int16Type, pa.types.is_int16),
        (Int32Type, pa.types.is_int32),
        (Int64Type, pa.types.is_int64),
        (UInt8Type, pa.types.is_uint8),
        (UInt16Type, pa.types.is_uint16),
        (UInt32Type, pa.types.is_uint32),
        (UInt64Type, pa.types.is_uint64),
        (Float16Type, pa.types.is_float16),
        (Float32Type, pa.types.is_float32),
        (Float64Type, pa.types.is_float64),
    ],
)
def test_primitive_arrow_type(
    dtype: PrimitiveType, arrow_type_checker: Callable[[pa.DataType], bool]
) -> None:
    assert arrow_type_checker(dtype.arrow_type)


class TestSequenceType:
    @pytest.mark.parametrize("length", [UNDEFINED_SEQUENCE_LENGTH, 5, 15])
    @pytest.mark.parametrize(
        "value_type,arrow_type_checker",
        [
            (BoolType, pa.types.is_boolean),
            (StringType, pa.types.is_string),
        ],
    )
    def test_arrow_type(
        self, length: int, value_type: Type, arrow_type_checker: Callable[[pa.DataType], bool]
    ) -> None:
        # create the sequence type
        sequence_type = SequenceType(value_type=value_type, length=length)

        # check the corresponding arrow type
        if length != UNDEFINED_SEQUENCE_LENGTH:
            assert pa.types.is_fixed_size_list(sequence_type.arrow_type)
            assert sequence_type.arrow_type.list_size == length
        else:
            assert pa.types.is_list(sequence_type.arrow_type)

        # check the value type
        assert arrow_type_checker(sequence_type.arrow_type.value_type)

    @pytest.mark.parametrize("length", [UNDEFINED_SEQUENCE_LENGTH, 5, 15])
    def test_get_item(self, length: int) -> None:
        # create the sequence type
        sequence_type = SequenceType(value_type=BoolType, length=length)

        # index all valid entries, limit to the first 100 entries for undefined sequence length
        for i in range(length if length != UNDEFINED_SEQUENCE_LENGTH else 100):
            assert sequence_type[i] == BoolType

        # only for fixed-sized sequence types
        if length < UNDEFINED_SEQUENCE_LENGTH:
            # raise index error when index is out of bounds
            with pytest.raises(IndexError):
                sequence_type[length]

    @pytest.mark.parametrize(
        "length,idx,expected_length",
        [
            (UNDEFINED_SEQUENCE_LENGTH, slice(0, 10, 1), UNDEFINED_SEQUENCE_LENGTH),
            (UNDEFINED_SEQUENCE_LENGTH, slice(5, 15, 1), UNDEFINED_SEQUENCE_LENGTH),
            (10, slice(10, 10, 1), 0),
            (10, slice(0, 10, 1), 10),
            (10, slice(5, 10, 1), 5),
            (10, slice(0, 5, 1), 5),
            (10, slice(2, 7, 1), 5),
            (10, slice(0, 10, 2), 5),
            (10, slice(5, 10, 2), 3),
            (10, slice(0, 5, 2), 3),
            (10, slice(2, 7, 2), 3),
        ],
    )
    def test_get_slice(self, length: int, idx: slice, expected_length: int) -> None:
        # create the sequence type and apply the index
        sequence_type = SequenceType(value_type=BoolType, length=length)
        indexed_type = sequence_type[idx]

        # check the indexed type
        assert isinstance(indexed_type, SequenceType)
        assert indexed_type.value_type == BoolType

        # check the length
        assert indexed_type.length == expected_length

    def test_sequence_equals(self) -> None:
        assert SequenceType(value_type=BoolType) == SequenceType(value_type=BoolType)
        assert SequenceType(value_type=BoolType, length=10) == SequenceType(
            value_type=BoolType, length=10
        )

        assert SequenceType(value_type=BoolType) != SequenceType(value_type=Int16Type)
        assert SequenceType(value_type=BoolType, length=10) != SequenceType(
            value_type=BoolType, length=5
        )

        assert SequenceType(value_type=BoolType) != "DummyString"


class TestMappingType:
    @pytest.mark.parametrize(
        "dtype, arrow_type",
        [
            (
                MappingType((("fieldA", BoolType), ("fieldB", Int16Type))),
                pa.struct([("fieldA", pa.bool_()), ("fieldB", pa.int16())]),
            )
        ],
    )
    def test_arrow_type(self, dtype: MappingType, arrow_type: pa.DataType) -> None:
        # check the arrow type
        assert pa.types.is_struct(dtype.arrow_type)
        assert dtype.arrow_type == arrow_type
        # check the arrow schema
        assert isinstance(dtype.arrow_schema, pa.Schema)
        assert pa.struct(dtype.arrow_schema) == arrow_type

    def test_mapping_interface(self) -> None:
        dtype = MappingType((("fieldA", BoolType), ("fieldB", Int16Type)))
        assert len(dtype) == 2
        assert set(dtype.keys()) == {"fieldA", "fieldB"}
        assert set(dtype.values()) == {BoolType, Int16Type}
        assert dtype["fieldA"] == BoolType
        assert dtype["fieldB"] == Int16Type

    def test_mapping_equals(self) -> None:
        assert MappingType((("fieldA", BoolType),)) == MappingType((("fieldA", BoolType),))
        assert MappingType((("fieldA", BoolType),)) != MappingType((("fieldA", Int16Type),))
        assert MappingType((("fieldA", BoolType),)) != MappingType((("fieldB", BoolType),))
        assert MappingType((("fieldA", BoolType),)) != "DummyString"

    def test_from_dict(self) -> None:
        dtype = MappingType((("fieldA", BoolType), ("fieldB", Int16Type)))
        assert dtype == MappingType.construct(dict(dtype))


@pytest.mark.parametrize(
    "dtype",
    [
        BoolType,
        StringType,
        Int8Type,
        Int16Type,
        Int32Type,
        Int64Type,
        UInt8Type,
        UInt16Type,
        UInt32Type,
        UInt64Type,
        Float16Type,
        Float32Type,
        Float64Type,
        SequenceType(BoolType),
        SequenceType(BoolType, length=10),
        MappingType.construct({"field": BoolType}),
        MappingType.construct({"field": SequenceType(BoolType)}),
    ],
)
def test_type_serialization(dtype: Type) -> None:
    assert build_type_from_dict(dtype.to_dict()) == dtype
