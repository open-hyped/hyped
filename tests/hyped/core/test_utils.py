from typing import Annotated, Any
from unittest.mock import MagicMock

import datasets
import pytest
from datasets.features.features import FeatureType

from hyped.core.features.dtypes import Int32Type, MappingType, SequenceType, Type
from hyped.core.features.features import build_feature_from_annotation
from hyped.core.utils import build_annotation_from_dtype, map_recursive, validate_hf_feature
from hyped.typing import Int32, Int64, Len, Mapping, Sequence


def test_map_recursive() -> None:
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


@pytest.mark.parametrize(
    "dtype",
    [
        Int32Type,
        SequenceType(Int32Type),
        SequenceType(SequenceType(Int32Type)),
        SequenceType(Int32Type, 5),
        SequenceType(SequenceType(Int32Type), 5),
        SequenceType(SequenceType(Int32Type, 10), 5),
        MappingType.construct({"fieldA": Int32Type, "fieldB": Int32Type}),
        MappingType.construct({"fieldA": SequenceType(Int32Type), "fieldB": Int32Type}),
        MappingType.construct({"fieldA": SequenceType(Int32Type, 5), "fieldB": Int32Type}),
        MappingType.construct(
            {"fieldA": MappingType.construct({"fieldA": Int32Type}), "fieldB": Int32Type}
        ),
    ],
)
def test_build_annotation_from_dtype(dtype: Type) -> None:
    # test reconstruct dtype from annotation
    annotation = build_annotation_from_dtype(dtype)
    assert dtype == build_feature_from_annotation(annotation).dtype


@pytest.mark.parametrize(
    "feature, annotation, raises_error",
    [
        (datasets.Value("int32"), Int32, False),
        (datasets.Value("int64"), Int64, False),
        (datasets.Value("int32"), Int64, True),
        (datasets.Value("int64"), Int32, True),
        (datasets.Sequence(datasets.Value("int32")), Sequence[Int32], False),
        (datasets.Sequence(datasets.Value("int32"), 5), Sequence[Int32], False),
        (datasets.Sequence(datasets.Value("int32"), 5), Annotated[Sequence[Int32], Len(5)], False),
        (datasets.Sequence(datasets.Value("int32"), 5), Annotated[Sequence[Int32], Len(10)], True),
        (
            datasets.Features(
                {"fieldA": datasets.Value("int32"), "fieldB": datasets.Value("int64")}
            ),
            type("Mapping", (Mapping,), {"__annotations__": {"fieldA": Int32, "fieldB": Int64}}),
            False,
        ),
        (
            datasets.Features(
                {"fieldA": datasets.Value("int32"), "fieldB": datasets.Value("int64")}
            ),
            type("Mapping", (Mapping,), {"__annotations__": {"fieldA": Int64, "fieldB": Int64}}),
            True,
        ),
    ],
)
def test_validate_hf_feature(feature: FeatureType, annotation: Any, raises_error: bool) -> None:
    if raises_error:
        with pytest.raises(RuntimeError):
            validate_hf_feature(feature, annotation)
    else:
        validate_hf_feature(feature, annotation)
