from typing import Callable

import datasets
import pyarrow as pa
import pytest
from datasets.features.features import FeatureType

from hyped.core.features.dtypes import (
    UNDEFINED_SEQUENCE_LENGTH,
    BoolType,
    ClassLabelType,
    DType,
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
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    build_dtype_from_arrow_type,
    build_dtype_from_dict,
    build_dtype_from_hf_feature,
    build_dtype_from_python_object,
    cast_dtype,
    common_dtype,
    is_dtype_subset,
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
        (ClassLabelType(names=tuple()), pa.types.is_int64),
    ],
)
def test_primitive_arrow_type(
    dtype: PrimitiveType, arrow_type_checker: Callable[[pa.DataType], bool]
) -> None:
    assert arrow_type_checker(dtype.arrow_type)


@pytest.mark.parametrize(
    "dtype, hf_feature",
    [
        (BoolType, datasets.Value("bool")),
        (Int8Type, datasets.Value("int8")),
        (Int16Type, datasets.Value("int16")),
        (Int32Type, datasets.Value("int32")),
        (Int64Type, datasets.Value("int64")),
        (UInt8Type, datasets.Value("uint8")),
        (UInt16Type, datasets.Value("uint16")),
        (UInt32Type, datasets.Value("uint32")),
        (UInt64Type, datasets.Value("uint64")),
        (Float16Type, datasets.Value("float16")),
        (Float32Type, datasets.Value("float32")),
        (Float64Type, datasets.Value("float64")),
        (ClassLabelType(names=("A", "B")), datasets.ClassLabel(names=["A", "B"])),
        (SequenceType(BoolType), datasets.Sequence(datasets.Value("bool"))),
        (SequenceType(BoolType, 10), datasets.Sequence(datasets.Value("bool"), length=10)),
        (
            MappingType.construct({"field": BoolType}),
            datasets.Features({"field": datasets.Value("bool")}),
        ),
    ],
)
def test_hf_feature(dtype: DType, hf_feature: FeatureType) -> None:
    assert dtype.hf_feature == hf_feature


def test_sequence_of_mapping_hf_feature_is_row_oriented() -> None:
    """Regression: a sequence of mappings must stay row-oriented through ``hf_feature``.

    ``datasets.Sequence`` transposes a struct value type into a columnar "struct of lists",
    which disagreed with ``SequenceType.arrow_type`` (``list<struct>``) and made the
    ``hf_feature`` round-trip lossy for sequences of mappings.
    """
    dtype = SequenceType(MappingType.construct({"a": Int64Type, "b": BoolType}))

    # arrow_type is row-oriented: list<struct<...>>
    assert pa.types.is_list(dtype.arrow_type)
    assert pa.types.is_struct(dtype.arrow_type.value_type)

    # the hf_feature round-trip must be lossless (not collapse into a columnar mapping)
    assert build_dtype_from_hf_feature(dtype.hf_feature) == dtype


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
        self, length: int, value_type: DType, arrow_type_checker: Callable[[pa.DataType], bool]
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
        ClassLabelType(names=("A", "B")),
        SequenceType(BoolType),
        SequenceType(BoolType, length=10),
        MappingType.construct({"field": BoolType}),
        MappingType.construct({"field": SequenceType(BoolType)}),
    ],
)
def test_type_serialization(dtype: DType) -> None:
    assert build_dtype_from_dict(dtype.to_dict()) == dtype


@pytest.mark.parametrize(
    "src_dtype, tgt_dtype, result_dtype, raises_error",
    [
        (BoolType, BoolType, BoolType, False),
        (Int16Type, Int32Type, Int32Type, False),
        (Int32Type, Int32Type, Int32Type, False),
        (Int32Type, Float32Type, Float32Type, False),
        (Float32Type, Int32Type, Int32Type, False),
        (ClassLabelType(names=tuple()), Int32Type, Int32Type, False),
        (Int32Type, ClassLabelType(names=tuple()), None, True),
        (ClassLabelType(names=("A", "B")), ClassLabelType(names=("A", "B", "C")), None, True),
        (
            ClassLabelType(names=("A", "B")),
            ClassLabelType(names=("X", "Y")),
            ClassLabelType(names=("X", "Y")),
            False,
        ),
        (SequenceType(BoolType), SequenceType(BoolType), SequenceType(BoolType), False),
        (SequenceType(BoolType, 5), SequenceType(BoolType), SequenceType(BoolType, 5), False),
        (SequenceType(BoolType), SequenceType(BoolType, 5), SequenceType(BoolType, 5), False),
        (SequenceType(BoolType, 5), SequenceType(BoolType, 5), SequenceType(BoolType, 5), False),
        (SequenceType(BoolType, 5), SequenceType(BoolType, 10), None, True),
        (
            MappingType.construct({"field": BoolType}),
            MappingType.construct({"field": BoolType}),
            MappingType.construct({"field": BoolType}),
            False,
        ),
        (
            MappingType.construct({"fieldA": BoolType}),
            MappingType.construct({"fieldB": BoolType}),
            None,
            True,
        ),
        (SequenceType(BoolType, 5), MappingType.construct({"field": BoolType}), None, True),
        (
            MappingType.construct({"field": Int32Type, "other": BoolType}),
            MappingType.construct({"field": Int64Type}),
            MappingType.construct({"field": Int64Type, "other": BoolType}),
            False,
        ),
    ],
)
def test_cast_dtype(
    src_dtype: DType, tgt_dtype: DType, result_dtype: None | DType, raises_error: bool
) -> None:
    if raises_error:
        with pytest.raises(RuntimeError):
            cast_dtype(src_dtype, tgt_dtype)
    else:
        # cast output should be target data type
        assert cast_dtype(src_dtype, tgt_dtype) == result_dtype


@pytest.mark.parametrize(
    "dtypes, result_dtype, raises_error",
    [
        # trivial case
        ((BoolType,), BoolType, False),
        # combining primitives
        ((BoolType, BoolType), BoolType, False),
        ((BoolType, Int8Type), Int8Type, False),
        ((BoolType, UInt8Type), UInt8Type, False),
        ((BoolType, Int16Type), Int16Type, False),
        ((BoolType, UInt16Type), UInt16Type, False),
        ((BoolType, Int32Type), Int32Type, False),
        ((BoolType, UInt32Type), UInt32Type, False),
        ((BoolType, Int64Type), Int64Type, False),
        ((BoolType, UInt64Type), UInt64Type, False),
        ((BoolType, Float16Type), Float16Type, False),
        ((BoolType, Float32Type), Float32Type, False),
        ((BoolType, Float64Type), Float64Type, False),
        ((BoolType, StringType), StringType, False),
        ((Int8Type, BoolType), Int8Type, False),
        ((Int8Type, Int8Type), Int8Type, False),
        ((Int8Type, UInt8Type), Int16Type, False),
        ((Int8Type, Int16Type), Int16Type, False),
        ((Int8Type, UInt16Type), Int32Type, False),
        ((Int8Type, Int32Type), Int32Type, False),
        ((Int8Type, UInt32Type), Int64Type, False),
        ((Int8Type, Int64Type), Int64Type, False),
        ((Int8Type, UInt64Type), Int64Type, False),
        ((Int8Type, Float16Type), Float16Type, False),
        ((Int8Type, Float32Type), Float32Type, False),
        ((Int8Type, Float64Type), Float64Type, False),
        ((Int8Type, StringType), StringType, False),
        ((UInt8Type, BoolType), UInt8Type, False),
        ((UInt8Type, Int8Type), Int16Type, False),
        ((UInt8Type, UInt8Type), UInt8Type, False),
        ((UInt8Type, Int16Type), Int16Type, False),
        ((UInt8Type, UInt16Type), UInt16Type, False),
        ((UInt8Type, Int32Type), Int32Type, False),
        ((UInt8Type, UInt32Type), UInt32Type, False),
        ((UInt8Type, Int64Type), Int64Type, False),
        ((UInt8Type, UInt64Type), UInt64Type, False),
        ((UInt8Type, Float16Type), Float16Type, False),
        ((UInt8Type, Float32Type), Float32Type, False),
        ((UInt8Type, Float64Type), Float64Type, False),
        ((UInt8Type, StringType), StringType, False),
        ((Int16Type, BoolType), Int16Type, False),
        ((Int16Type, Int8Type), Int16Type, False),
        ((Int16Type, UInt8Type), Int16Type, False),
        ((Int16Type, Int16Type), Int16Type, False),
        ((Int16Type, UInt16Type), Int32Type, False),
        ((Int16Type, Int32Type), Int32Type, False),
        ((Int16Type, UInt32Type), Int64Type, False),
        ((Int16Type, Int64Type), Int64Type, False),
        ((Int16Type, UInt64Type), Int64Type, False),
        ((Int16Type, Float16Type), Float16Type, False),
        ((Int16Type, Float32Type), Float32Type, False),
        ((Int16Type, Float64Type), Float64Type, False),
        ((Int16Type, StringType), StringType, False),
        ((UInt16Type, BoolType), UInt16Type, False),
        ((UInt16Type, Int8Type), Int32Type, False),
        ((UInt16Type, UInt8Type), UInt16Type, False),
        ((UInt16Type, Int16Type), Int32Type, False),
        ((UInt16Type, UInt16Type), UInt16Type, False),
        ((UInt16Type, Int32Type), Int32Type, False),
        ((UInt16Type, UInt32Type), UInt32Type, False),
        ((UInt16Type, Int64Type), Int64Type, False),
        ((UInt16Type, UInt64Type), UInt64Type, False),
        ((UInt16Type, Float16Type), Float16Type, False),
        ((UInt16Type, Float32Type), Float32Type, False),
        ((UInt16Type, Float64Type), Float64Type, False),
        ((UInt16Type, StringType), StringType, False),
        ((Int32Type, BoolType), Int32Type, False),
        ((Int32Type, Int8Type), Int32Type, False),
        ((Int32Type, UInt8Type), Int32Type, False),
        ((Int32Type, Int16Type), Int32Type, False),
        ((Int32Type, UInt16Type), Int32Type, False),
        ((Int32Type, Int32Type), Int32Type, False),
        ((Int32Type, UInt32Type), Int64Type, False),
        ((Int32Type, Int64Type), Int64Type, False),
        ((Int32Type, UInt64Type), Int64Type, False),
        ((Int32Type, Float16Type), Float16Type, False),
        ((Int32Type, Float32Type), Float32Type, False),
        ((Int32Type, Float64Type), Float64Type, False),
        ((Int32Type, StringType), StringType, False),
        ((UInt32Type, BoolType), UInt32Type, False),
        ((UInt32Type, Int8Type), Int64Type, False),
        ((UInt32Type, UInt8Type), UInt32Type, False),
        ((UInt32Type, Int16Type), Int64Type, False),
        ((UInt32Type, UInt16Type), UInt32Type, False),
        ((UInt32Type, Int32Type), Int64Type, False),
        ((UInt32Type, UInt32Type), UInt32Type, False),
        ((UInt32Type, Int64Type), Int64Type, False),
        ((UInt32Type, UInt64Type), UInt64Type, False),
        ((UInt32Type, Float16Type), Float16Type, False),
        ((UInt32Type, Float32Type), Float32Type, False),
        ((UInt32Type, Float64Type), Float64Type, False),
        ((UInt32Type, StringType), StringType, False),
        ((Int64Type, BoolType), Int64Type, False),
        ((Int64Type, Int8Type), Int64Type, False),
        ((Int64Type, UInt8Type), Int64Type, False),
        ((Int64Type, Int16Type), Int64Type, False),
        ((Int64Type, UInt16Type), Int64Type, False),
        ((Int64Type, Int32Type), Int64Type, False),
        ((Int64Type, UInt32Type), Int64Type, False),
        ((Int64Type, Int64Type), Int64Type, False),
        ((Int64Type, UInt64Type), Int64Type, False),
        ((Int64Type, Float16Type), Float16Type, False),
        ((Int64Type, Float32Type), Float32Type, False),
        ((Int64Type, Float64Type), Float64Type, False),
        ((Int64Type, StringType), StringType, False),
        ((UInt64Type, BoolType), UInt64Type, False),
        ((UInt64Type, Int8Type), Int64Type, False),
        ((UInt64Type, UInt8Type), UInt64Type, False),
        ((UInt64Type, Int16Type), Int64Type, False),
        ((UInt64Type, UInt16Type), UInt64Type, False),
        ((UInt64Type, Int32Type), Int64Type, False),
        ((UInt64Type, UInt32Type), UInt64Type, False),
        ((UInt64Type, Int64Type), Int64Type, False),
        ((UInt64Type, UInt64Type), UInt64Type, False),
        ((UInt64Type, Float16Type), Float16Type, False),
        ((UInt64Type, Float32Type), Float32Type, False),
        ((UInt64Type, Float64Type), Float64Type, False),
        ((UInt64Type, StringType), StringType, False),
        ((Float16Type, BoolType), Float16Type, False),
        ((Float16Type, Int8Type), Float16Type, False),
        ((Float16Type, UInt8Type), Float16Type, False),
        ((Float16Type, Int16Type), Float16Type, False),
        ((Float16Type, UInt16Type), Float16Type, False),
        ((Float16Type, Int32Type), Float16Type, False),
        ((Float16Type, UInt32Type), Float16Type, False),
        ((Float16Type, Int64Type), Float16Type, False),
        ((Float16Type, UInt64Type), Float16Type, False),
        ((Float16Type, Float16Type), Float16Type, False),
        ((Float16Type, Float32Type), Float32Type, False),
        ((Float16Type, Float64Type), Float64Type, False),
        ((Float16Type, StringType), StringType, False),
        ((Float32Type, BoolType), Float32Type, False),
        ((Float32Type, Int8Type), Float32Type, False),
        ((Float32Type, UInt8Type), Float32Type, False),
        ((Float32Type, Int16Type), Float32Type, False),
        ((Float32Type, UInt16Type), Float32Type, False),
        ((Float32Type, Int32Type), Float32Type, False),
        ((Float32Type, UInt32Type), Float32Type, False),
        ((Float32Type, Int64Type), Float32Type, False),
        ((Float32Type, UInt64Type), Float32Type, False),
        ((Float32Type, Float16Type), Float32Type, False),
        ((Float32Type, Float32Type), Float32Type, False),
        ((Float32Type, Float64Type), Float64Type, False),
        ((Float32Type, StringType), StringType, False),
        ((Float64Type, BoolType), Float64Type, False),
        ((Float64Type, Int8Type), Float64Type, False),
        ((Float64Type, UInt8Type), Float64Type, False),
        ((Float64Type, Int16Type), Float64Type, False),
        ((Float64Type, UInt16Type), Float64Type, False),
        ((Float64Type, Int32Type), Float64Type, False),
        ((Float64Type, UInt32Type), Float64Type, False),
        ((Float64Type, Int64Type), Float64Type, False),
        ((Float64Type, UInt64Type), Float64Type, False),
        ((Float64Type, Float16Type), Float64Type, False),
        ((Float64Type, Float32Type), Float64Type, False),
        ((Float64Type, Float64Type), Float64Type, False),
        ((Float64Type, StringType), StringType, False),
        ((StringType, BoolType), StringType, False),
        ((StringType, Int8Type), StringType, False),
        ((StringType, UInt8Type), StringType, False),
        ((StringType, Int16Type), StringType, False),
        ((StringType, UInt16Type), StringType, False),
        ((StringType, Int32Type), StringType, False),
        ((StringType, UInt32Type), StringType, False),
        ((StringType, Int64Type), StringType, False),
        ((StringType, UInt64Type), StringType, False),
        ((StringType, Float16Type), StringType, False),
        ((StringType, Float32Type), StringType, False),
        ((StringType, Float64Type), StringType, False),
        ((StringType, StringType), StringType, False),
        ((ClassLabelType(names=tuple()), Int64Type), Int64Type, False),
        ((ClassLabelType(names=tuple()), Int32Type), Int64Type, False),
        ((ClassLabelType(names=tuple()), StringType), StringType, False),
        # nested types
        (
            (
                SequenceType(BoolType, 5),
                SequenceType(Int32Type, 5),
                SequenceType(Int64Type, 5),
            ),
            SequenceType(Int64Type, 5),
            False,
        ),
        (
            (
                SequenceType(BoolType, 5),
                SequenceType(Int32Type, 3),
                SequenceType(Int64Type, 5),
            ),
            SequenceType(Int64Type, UNDEFINED_SEQUENCE_LENGTH),
            False,
        ),
        (
            (
                MappingType.construct({"field": BoolType}),
                MappingType.construct({"field": BoolType}),
            ),
            MappingType.construct({"field": BoolType}),
            False,
        ),
        (
            (
                MappingType.construct({"fieldA": BoolType}),
                MappingType.construct({"fieldB": BoolType}),
            ),
            None,
            True,
        ),
        (
            (
                SequenceType(BoolType, 5),
                MappingType.construct({"fieldB": BoolType}),
            ),
            None,
            True,
        ),
    ],
)
def test_common_dtype(dtypes: tuple[DType], result_dtype: None | DType, raises_error: bool) -> None:
    if raises_error:
        with pytest.raises(RuntimeError):
            common_dtype(*dtypes)
    else:
        # cast output should be target data type
        assert common_dtype(*dtypes) == result_dtype


@pytest.mark.parametrize(
    "arrow_type, expected_output, expect_exception",
    [
        # Scalar type conversion
        (pa.int32(), Int32Type, False),
        # Unsupported type
        (pa.null(), None, True),
        # Struct type conversion
        (
            pa.struct([("field1", pa.int32()), ("field2", pa.float64())]),
            MappingType((("field1", Int32Type), ("field2", Float64Type))),
            False,
        ),
        (
            pa.struct([("field2", pa.int32()), ("field1", pa.float64())]),
            MappingType((("field2", Int32Type), ("field1", Float64Type))),
            False,
        ),
        # List type conversion
        (
            pa.list_(pa.int32()),
            SequenceType(value_type=Int32Type),
            False,
        ),
        # Fixed-Sized List type conversion
        (
            pa.list_(pa.int32(), 5),
            SequenceType(value_type=Int32Type, length=5),
            False,
        ),
    ],
)
def test_build_dtype_from_arrow_type(arrow_type, expected_output, expect_exception):
    if expect_exception:
        with pytest.raises(TypeError):
            build_dtype_from_arrow_type(arrow_type)
    else:
        result = build_dtype_from_arrow_type(arrow_type)
        assert result == expected_output


@pytest.mark.parametrize(
    "hf_feature, expected_output, expect_exception",
    [
        # Value type conversion
        (datasets.Value("int32"), Int32Type, False),
        # Class label conversion
        (datasets.ClassLabel(names=("A", "B")), ClassLabelType(names=("A", "B")), False),
        # Unsupported feature type
        (object(), None, True),
        # Features type conversion
        (
            datasets.Features(
                {"field1": datasets.Value("int32"), "field2": datasets.Value("float64")}
            ),
            MappingType((("field1", Int32Type), ("field2", Float64Type))),
            False,
        ),
        (
            datasets.Features(
                {"field2": datasets.Value("int32"), "field1": datasets.Value("float64")}
            ),
            MappingType((("field2", Int32Type), ("field1", Float64Type))),
            False,
        ),
        # Sequence type conversion (fixed length)
        (
            datasets.Sequence(feature=datasets.Value("int32"), length=10),
            SequenceType(value_type=Int32Type, length=10),
            False,
        ),
        # Sequence type conversion (undefined length)
        (
            datasets.Sequence(feature=datasets.Value("float64")),
            SequenceType(value_type=Float64Type, length=UNDEFINED_SEQUENCE_LENGTH),
            False,
        ),
    ],
)
def test_build_dtype_from_hf_feature(hf_feature, expected_output, expect_exception):
    if expect_exception:
        with pytest.raises(TypeError):
            build_dtype_from_hf_feature(hf_feature)
    else:
        result = build_dtype_from_hf_feature(hf_feature)
        assert result == expected_output


@pytest.mark.parametrize(
    "obj, expected_output, expect_exception, exception_type",
    [
        # Mapping type conversion
        (
            {"field1": 1, "field2": 3.14},
            MappingType.construct({"field1": Int32Type, "field2": Float64Type}),
            False,
            None,
        ),
        # Sequence type conversion (homogeneous list)
        ([1, 2, 3], SequenceType(value_type=Int32Type, length=3), False, None),
        # Sequence type conversion (heterogeneous list)
        ([1, 2.0, 3], None, True, RuntimeError),
        # Empty list
        ([], SequenceType(value_type=BoolType, length=0), False, None),
        # Primitive type conversion
        (42, Int32Type, False, None),
        (3.14, Float64Type, False, None),
        ("hello", StringType, False, None),
        (True, BoolType, False, None),
        # Unsupported type
        (object(), None, True, TypeError),
    ],
)
def test_build_dtype_from_python_object(obj, expected_output, expect_exception, exception_type):
    if expect_exception:
        with pytest.raises(exception_type):
            build_dtype_from_python_object(obj)
    else:
        result = build_dtype_from_python_object(obj)
        assert result == expected_output


@pytest.mark.parametrize(
    "dtype_a, dtype_b, expected_output",
    [
        # Scalar types (equal)
        (Int32Type, Int32Type, True),
        # Scalar types (not equal)
        (Int32Type, Float64Type, False),
        # Mapping types (subset)
        (
            MappingType.construct({"field1": Int32Type}),
            MappingType.construct({"field1": Int32Type, "field2": Float64Type}),
            True,
        ),
        # Mapping types (not a subset)
        (
            MappingType.construct({"field1": Int32Type, "field3": BoolType}),
            MappingType.construct({"field1": Int32Type, "field2": Float64Type}),
            False,
        ),
        # Sequence types (subset)
        (
            SequenceType(value_type=Int32Type, length=3),
            SequenceType(value_type=Int32Type, length=3),
            True,
        ),
        # Sequence types (length mismatch)
        (
            SequenceType(value_type=Int32Type, length=3),
            SequenceType(value_type=Int32Type, length=5),
            False,
        ),
        # Sequence types (value type mismatch)
        (
            SequenceType(value_type=Int32Type, length=3),
            SequenceType(value_type=Float64Type, length=3),
            False,
        ),
        # Complex nested types (subset)
        (
            MappingType.construct(
                {
                    "field1": Int32Type,
                    "field2": SequenceType(value_type=Float64Type, length=2),
                }
            ),
            MappingType.construct(
                {
                    "field1": Int32Type,
                    "field2": SequenceType(value_type=Float64Type, length=2),
                    "field3": BoolType,
                }
            ),
            True,
        ),
        # Complex nested types (not a subset)
        (
            MappingType.construct(
                {
                    "field1": Int32Type,
                    "field2": SequenceType(value_type=BoolType, length=2),
                }
            ),
            MappingType.construct(
                {
                    "field1": Int32Type,
                    "field2": SequenceType(value_type=Float64Type, length=2),
                    "field3": BoolType,
                }
            ),
            False,
        ),
    ],
)
def test_is_dtype_subset(dtype_a, dtype_b, expected_output):
    result = is_dtype_subset(dtype_a, dtype_b)
    assert result == expected_output
