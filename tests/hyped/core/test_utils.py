from unittest.mock import MagicMock

import datasets
import pyarrow as pa
import pytest

from hyped.core.features.types import (
    UNDEFINED_SEQUENCE_LENGTH,
    BoolType,
    Float64Type,
    Int32Type,
    MappingType,
    SequenceType,
    StringType,
)
from hyped.core.utils import (
    build_dtype_from_arrow_type,
    build_dtype_from_hf_feature,
    build_dtype_from_python_object,
    is_dtype_subset,
    map_recursive,
)


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
            MappingType.from_dict({"field1": Int32Type, "field2": Float64Type}),
            False,
        ),
        # List type conversion
        (
            pa.list_(pa.int32()),
            SequenceType(value_type=Int32Type),
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
        # Unsupported feature type
        (object(), None, True),
        # Features type conversion
        (
            datasets.Features(
                {"field1": datasets.Value("int32"), "field2": datasets.Value("float64")}
            ),
            MappingType.from_dict({"field1": Int32Type, "field2": Float64Type}),
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
            MappingType.from_dict({"field1": Int32Type, "field2": Float64Type}),
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
            MappingType.from_dict({"field1": Int32Type}),
            MappingType.from_dict({"field1": Int32Type, "field2": Float64Type}),
            True,
        ),
        # Mapping types (not a subset)
        (
            MappingType.from_dict({"field1": Int32Type, "field3": BoolType}),
            MappingType.from_dict({"field1": Int32Type, "field2": Float64Type}),
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
            MappingType.from_dict(
                {
                    "field1": Int32Type,
                    "field2": SequenceType(value_type=Float64Type, length=2),
                }
            ),
            MappingType.from_dict(
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
            MappingType.from_dict(
                {
                    "field1": Int32Type,
                    "field2": SequenceType(value_type=BoolType, length=2),
                }
            ),
            MappingType.from_dict(
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


def test_map_recursive():
    # Define a nested structure with various levels of dictionaries and lists
    nested_obj = {
        "key1": [1, 2, {"key2": 3}],
        "key3": {"key4": [4, 5], "key5": 6},
    }

    # Mock function to apply
    mock_fn = MagicMock(side_effect=lambda path, x: x)

    # Apply map_recursive
    result = map_recursive(mock_fn, nested_obj)

    # Verify the structure remains unchanged (since fn returns x unchanged)
    assert result == nested_obj

    # Verify the mock function was called for every element in the nested object
    expected_calls = [
        ((tuple(), nested_obj),),
        # Calls for top-level keys
        ((("key1",), [1, 2, {"key2": 3}]),),
        ((("key3",), {"key4": [4, 5], "key5": 6}),),
        # Calls for elements inside key1
        ((("key1", 0), 1),),
        ((("key1", 1), 2),),
        ((("key1", 2), {"key2": 3}),),
        ((("key1", 2, "key2"), 3),),
        # Calls for elements inside key3
        ((("key3", "key4"), [4, 5]),),
        ((("key3", "key4", 0), 4),),
        ((("key3", "key4", 1), 5),),
        ((("key3", "key5"), 6),),
    ]

    mock_fn.assert_has_calls(expected_calls, any_order=True)

    # Verify the number of calls matches the number of elements in the structure
    assert mock_fn.call_count == len(expected_calls)
