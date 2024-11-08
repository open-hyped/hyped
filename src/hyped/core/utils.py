from __future__ import annotations

from typing import Any, Callable, TypeAlias, TypeVar

import datasets
import pyarrow as pa
from datasets.features.features import FeatureType

from hyped.common.typing import ArrowType

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
        seq (Sequence | list | tuple): sequence to get the length of

    Returns:
        length (int):
            the length of the given sequence. Returns -1 for
            sequences of undefined length
    """
    assert isinstance(seq, (datasets.Sequence, list, tuple))
    return seq.length if isinstance(seq, datasets.Sequence) else -1


def get_hf_sequence_feature(seq: datasets.Sequence | list | tuple) -> FeatureType:
    """Get the item feature type of a given sequence feature.

    Arguments:
        seq (Sequence | list | tuple):
            sequence to get the item feature type of

    Returns:
        feature (FeatureType):
            the item feature type of the sequence
    """
    assert isinstance(seq, (datasets.Sequence, list, tuple))
    return seq.feature if isinstance(seq, datasets.Sequence) else seq[0]


def build_dtype_from_arrow_type(arrow_type: ArrowType) -> Type:
    if pa.types.is_struct(arrow_type):
        fields = map(arrow_type.field, range(arrow_type.num_fields))
        fields = {field.name: build_dtype_from_arrow_type(field.type) for field in fields}
        return MappingType.from_dict(fields)

    elif pa.types.is_list(arrow_type):
        return SequenceType(
            value_type=build_dtype_from_arrow_type(arrow_type.value_type),
        )

    if isinstance(arrow_type, ArrowType):
        return ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING[str(arrow_type)]

    raise TypeError()


def build_dtype_from_hf_feature(feature: FeatureType) -> Type:
    if isinstance(feature, datasets.Features):
        # get all field types
        fields = {key: build_dtype_from_hf_feature(field) for key, field in feature.items()}
        # create a mapping type from the fields
        return MappingType.from_dict(fields)

    if isinstance(feature, datasets.Sequence):
        # get the value type and sequence length
        value_type = get_hf_sequence_feature(feature)
        length = get_hf_sequence_length(feature)
        # build the sequence type
        return SequenceType(
            value_type=build_dtype_from_hf_feature(value_type),
            length=UNDEFINED_SEQUENCE_LENGTH if length == -1 else length,
        )

    elif isinstance(feature, datasets.Value):
        # get the arrow type for the value
        packed = datasets.Features({"field": feature})
        arrow_type = packed.arrow_schema.field("field").type
        # find the data type to the arrow type
        return ARROW_SCALAR_TYPE_TO_DTYPE_MAPPING[str(arrow_type)]

    raise TypeError(feature)


def build_dtype_from_object(obj: Any) -> Type:
    if isinstance(obj, dict):
        return MappingType.from_dict(
            {key: build_dtype_from_object(val) for key, val in obj.items()}
        )

    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            # build the dtype for each item in the sequence
            dtype, *others = list(map(build_dtype_from_object, obj))
            # make sure all values in the list are of the same type
            if any(dtype != other for other in others):
                raise RuntimeError()  # TODO: error message

        else:
            # default dtype for empty sequences
            dtype = BoolType

        return SequenceType(value_type=dtype, length=len(obj))

    elif isinstance(obj, tuple(PYTHON_PRIMITIVE_TO_DTYPE_MAPPING.keys())):
        return PYTHON_PRIMITIVE_TO_DTYPE_MAPPING[type(obj)]

    else:
        raise TypeError(obj)  # TODO: error message unsupported type


def is_dtype_subset(dtype_a: Type, dtype_b: Type) -> bool:
    """
    Recursively checks if type_a is a subset of type_b.

    Args:
        type_a (Type): The type that should be a subset.
        type_b (Type): The type that should be a superset.

    Returns:
        bool: True if type_a is a subset of type_b, False otherwise.
    """

    if isinstance(dtype_a, MappingType) and isinstance(dtype_b, MappingType):
        # test all fields of type A against the fields in type B
        return all(
            (key in dtype_b) and is_dtype_subset(dtype, dtype_b[key])
            for key, dtype in dtype_a.items()
        )

    elif isinstance(dtype_a, SequenceType) and isinstance(dtype_b, SequenceType):
        # TODO: support length <undefined> == n and a.length <= b.length
        return (dtype_a.length == dtype_b.length) and is_dtype_subset(
            dtype_a.arrow_type, dtype_b.arrow_type
        )

    return dtype_a == dtype_b


def map_recursive(
    fn: Callable[[NestedType[Any]], None | NestedType[Any]],
    obj: NestedType[Any],
    path: tuple[str | int] = (),
) -> NestedType[Any]:
    # apply the function recursively on all items of the nested object
    obj = (
        {key: map_recursive(fn, val, path + (key,)) for key, val in obj.items()}
        if isinstance(obj, dict)
        else type(obj)(map_recursive(fn, val, path + (i,)) for i, val in enumerate(obj))
        if isinstance(obj, (list, tuple))
        else obj
    )
    # apply the function on the full object
    out_obj = fn(path, obj)
    return out_obj if out_obj is not None else obj
