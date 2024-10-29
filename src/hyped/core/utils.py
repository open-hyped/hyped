from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import datasets
import numpy as np
import numpy.typing
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

ARROW_SCALAR_TYPE_TO_FEATURE_MAPPING: dict[str, Type] = {
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


def get_sequence_length(seq: datasets.Sequence | list | tuple) -> int:
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


def get_sequence_feature(seq: datasets.Sequence | list | tuple) -> FeatureType:
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
        return ARROW_SCALAR_TYPE_TO_FEATURE_MAPPING[str(arrow_type)]

    raise TypeError()


def build_dtype_from_hf_feature(feature: FeatureType) -> Type:
    if isinstance(feature, datasets.Features):
        # get all field types
        fields = {key: build_dtype_from_hf_feature(field) for key, field in feature.items()}
        # create a mapping type from the fields
        return MappingType.from_dict(fields)

    if isinstance(feature, datasets.Sequence):
        # get the value type and sequence length
        value_type = get_sequence_feature(feature)
        length = get_sequence_length(feature)
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
        return ARROW_SCALAR_TYPE_TO_FEATURE_MAPPING[str(arrow_type)]

    raise TypeError(feature)


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
        # TODO: support length <undefined> == n
        return (dtype_a.length == dtype_b.length) and is_dtype_subset(
            dtype_a.arrow_type, dtype_b.arrow_type
        )

    return dtype_a == dtype_b


@dataclass
class ArrowIndexingWrapper(object):
    # TODO: docstring: this wrapper allows arrow arrays to be passed to feature key
    array: pa.Array
    _buckets: None | list[numpy.typing.NDArray] = None

    def __getitem__(self, key: str | int | slice) -> "ArrowIndexingWrapper":
        if isinstance(key, str):
            array = pa.compute.struct_field(self.array, key)
            return ArrowIndexingWrapper(array)

        elif isinstance(key, int):
            array = pa.compute.list_element(self.array, key)
            return ArrowIndexingWrapper(array)

        elif isinstance(key, slice):
            # cannot overwrite buckets
            assert self._buckets is None

            # shorthand to array
            array = self.array

            # prepare the slice, cannot use .indices here because we don't know the length
            stop = key.stop
            start = key.start if key.start is not None else 0
            step = key.step if key.step is not None else 1

            # list_slice doesn't support negative values as stop index
            if key.stop < 0:
                raise NotImplementedError()

            if not ((stop is None) and (start == 0) and (step == 1)):
                # get the list slice but only if there are actually values being omitted be
                # slicing, otherwise (i.e. slice(None)) just keep the full list
                array = pa.compute.list_slice(array, start, stop, step)

            # flatten the list for further processing
            flat_array = pa.compute.list_flatten(array)
            # build information needed to invert the flatten operation
            # specifically a bucket represents a single list in the nested structure
            # and contains all indices of the list w.r.t. the flattened list
            parent_index = pa.compute.list_parent_indices(array).to_numpy()
            self._buckets = [np.nonzero(parent_index == j)[0] for j in range(len(array))]

            return [ArrowIndexingWrapper(flat_array)]

    def __slice_constructor__(self, items: Iterable[ArrowIndexingWrapper]) -> ArrowIndexingWrapper:
        # callback used by feature key allows for custom construction from slice operation

        # get the flat array from the items
        flat_array = next(iter(items)).array
        assert self._buckets is not None
        # unflatten the array using the buckets
        array = [pa.compute.take(flat_array, idx) for idx in self._buckets]
        array = pa.array(array, type=pa.list_(flat_array.type))
        # reset buckets
        self._buckets = None
        # return the nested array
        return ArrowIndexingWrapper(array)
