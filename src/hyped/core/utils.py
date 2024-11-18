"""Utility functions and type mappings used throughout the core module.

This module provides various helper functions for working with data types,
feature mappings, and nested structures in the core module.
"""
from __future__ import annotations

from typing import Any, Callable, TypeAlias, TypeVar

import datasets
import pyarrow as pa
from datasets.features.features import FeatureType

from .features.types import (
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
    SequenceType,
    StringType,
    Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
)

ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING: dict[str, Type] = {
    "bool": BoolType,
    "string": StringType,
    "int8": Int8Type,
    "int16": Int16Type,
    "int32": Int32Type,
    "int64": Int64Type,
    "uint8": UInt8Type,
    "uint16": UInt16Type,
    "uint32": UInt32Type,
    "uint64": UInt64Type,
    "halffloat": Float16Type,
    "float": Float32Type,
    "double": Float64Type,
}

PYTHON_PRIMITIVE_TO_DTYPE_MAPPING: dict[type, Type] = {
    bool: BoolType,
    str: StringType,
    int: Int64Type,
    float: Float64Type,
}

T = TypeVar("T")
NestedType: TypeAlias = dict[str, "NestedType"] | list["NestedType"] | tuple["NestedType"] | T


def get_hf_sequence_length(seq: datasets.Sequence | list | tuple) -> int:
    """Get the length of a given sequence feature.

    Arguments:
        seq (Sequence | list | tuple): Sequence to get the length of.

    Returns:
        int: The length of the given sequence. Returns -1 for sequences of undefined length.
    """
    assert isinstance(seq, (datasets.Sequence, list, tuple))
    return seq.length if isinstance(seq, datasets.Sequence) else -1


def get_hf_sequence_feature(seq: datasets.Sequence | list | tuple) -> FeatureType:
    """Get the item feature type of a given sequence feature.

    Arguments:
        seq (Sequence | list | tuple): Sequence to get the item feature type of.

    Returns:
        FeatureType: The item feature type of the sequence.
    """
    assert isinstance(seq, (datasets.Sequence, list, tuple))
    return seq.feature if isinstance(seq, datasets.Sequence) else seq[0]


def build_dtype_from_arrow_type(arrow_type: pa.DataType) -> Type:
    """Build a data type from a given Arrow type.

    Arguments:
        arrow_type (pa.DataType): The Arrow type to convert.

    Returns:
        Type: The corresponding data type.

    Raises:
        TypeError: If the Arrow type is unsupported.
    """
    if pa.types.is_struct(arrow_type):
        fields = map(arrow_type.field, range(arrow_type.num_fields))
        fields = {field.name: build_dtype_from_arrow_type(field.type) for field in fields}
        return MappingType.from_dict(fields)

    elif pa.types.is_list(arrow_type):
        return SequenceType(
            value_type=build_dtype_from_arrow_type(arrow_type.value_type),
        )

    if isinstance(arrow_type, pa.DataType):
        return ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING[str(arrow_type)]

    # TODO: error message
    raise TypeError()


def build_dtype_from_hf_feature(feature: FeatureType) -> Type:
    """Build a data type from a given Hugging Face feature.

    Arguments:
        feature (FeatureType): The Hugging Face feature to convert.

    Returns:
        Type: The corresponding data type.

    Raises:
        TypeError: If the feature type is unsupported.
    """
    if isinstance(feature, datasets.Features):
        fields = {key: build_dtype_from_hf_feature(field) for key, field in feature.items()}
        return MappingType.from_dict(fields)

    if isinstance(feature, datasets.Sequence):
        value_type = get_hf_sequence_feature(feature)
        length = get_hf_sequence_length(feature)
        return SequenceType(
            value_type=build_dtype_from_hf_feature(value_type),
            length=UNDEFINED_SEQUENCE_LENGTH if length == -1 else length,
        )

    elif isinstance(feature, datasets.Value):
        packed = datasets.Features({"field": feature})
        arrow_type = packed.arrow_schema.field("field").type
        return ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING[str(arrow_type)]

    # TODO: error message
    raise TypeError(feature)


def build_dtype_from_object(obj: Any) -> Type:
    """Build a data type from a Python object.

    Arguments:
        obj (Any): The object to derive the data type from.

    Returns:
        Type: The corresponding data type.

    Raises:
        TypeError: If the object type is unsupported.
        RuntimeError: If there are inconsistencies in list item types.
    """
    if isinstance(obj, dict):
        return MappingType.from_dict(
            {key: build_dtype_from_object(val) for key, val in obj.items()}
        )

    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            dtype, *others = list(map(build_dtype_from_object, obj))
            if any(dtype != other for other in others):
                raise RuntimeError()  # TODO: error message

        else:
            dtype = BoolType

        return SequenceType(value_type=dtype, length=len(obj))

    elif isinstance(obj, tuple(PYTHON_PRIMITIVE_TO_DTYPE_MAPPING.keys())):
        return PYTHON_PRIMITIVE_TO_DTYPE_MAPPING[type(obj)]

    else:
        raise TypeError(obj)  # TODO: error message unsupported type


def is_dtype_subset(dtype_a: Type, dtype_b: Type) -> bool:
    """Recursively checks if dtype_a is a subset of dtype_b.

    Arguments:
        dtype_a (Type): The type that should be a subset.
        dtype_b (Type): The type that should be a superset.

    Returns:
        bool: True if dtype_a is a subset of dtype_b, False otherwise.
    """
    if isinstance(dtype_a, MappingType) and isinstance(dtype_b, MappingType):
        return all(
            (key in dtype_b) and is_dtype_subset(dtype, dtype_b[key])
            for key, dtype in dtype_a.items()
        )

    elif isinstance(dtype_a, SequenceType) and isinstance(dtype_b, SequenceType):
        return (dtype_a.length == dtype_b.length) and is_dtype_subset(
            dtype_a.arrow_type, dtype_b.arrow_type
        )

    return dtype_a == dtype_b


def map_recursive(
    fn: Callable[[NestedType[Any]], None | NestedType[Any]],
    obj: NestedType[Any],
    path: tuple[str | int] = (),
) -> NestedType[Any]:
    """Apply a function recursively on a nested object.

    Arguments:
        fn (Callable[[NestedType[Any]], None | NestedType[Any]]): The function to apply.
        obj (NestedType[Any]): The nested object to apply the function to.
        path (tuple[str | int], optional): The current path in the object
            (default is an empty tuple).

    Returns:
        NestedType[Any]: The object after the function has been applied recursively.
    """
    obj = (
        {key: map_recursive(fn, val, path + (key,)) for key, val in obj.items()}
        if isinstance(obj, dict)
        else type(obj)(map_recursive(fn, val, path + (i,)) for i, val in enumerate(obj))
        if isinstance(obj, (list, tuple))
        else obj
    )
    out_obj = fn(path, obj)
    return out_obj if out_obj is not None else obj
