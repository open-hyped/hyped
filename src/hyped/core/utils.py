from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeVar

import datasets
import numpy as np
import numpy.typing
import pyarrow as pa
from datasets.features.features import FeatureType

from hyped.common.typing import ArrowType, NodeId

from .abstract import AbstractDataFlowGraph
from .features.factory import FeatureFactoryFromInstance, _DummyDataFlowGraph
from .features.features import (
    Mapping,
    Sequence,
    _Bool,
    _Feature,
    _Float16,
    _Float32,
    _Float64,
    _Int8,
    _Int16,
    _Int32,
    _Int64,
    _Primitive,
    _String,
    _UInt8,
    _UInt16,
    _UInt32,
    _UInt64,
)
from .features.reference import FeatureKey

ARROW_SCALAR_TYPE_TO_FEATURE_MAPPING: dict[str, _Primitive] = {
    "bool": _Bool,
    "string": _String,
    "int8": _Int8,
    "int16": _Int16,
    "int32": _Int32,
    "int64": _Int64,
    "uint8": _UInt8,
    "uint16": _UInt16,
    "uint32": _UInt32,
    "uint64": _UInt64,
    "halffloat": _Float16,
    "float": _Float32,
    "double": _Float64,
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


def _build_feature_type_from_arrow_type(pa_type: ArrowType) -> type:
    if isinstance(pa_type, pa.StructType):
        fields = map(pa_type.field, range(pa_type.num_fields))
        annotations = {
            field.name: _build_feature_type_from_arrow_type(field.type) for field in fields
        }
        # TODO: implement a dynamic mapping that doesn't rely on the annotations
        return type("Mapping", (Mapping,), {"__annotations__": annotations})

    if isinstance(pa_type, pa.ListType):
        return Sequence[_build_feature_type_from_arrow_type(pa_type.value_type)]

    if isinstance(pa_type, ArrowType):
        return ARROW_SCALAR_TYPE_TO_FEATURE_MAPPING[str(pa_type)]

    raise TypeError()


def _is_type_subset(type_a: pa.DataType, type_b: pa.DataType) -> bool:
    """
    Recursively checks if type_a is a subset of type_b.

    Args:
        type_a (pa.DataType): The type that should be a subset.
        type_b (pa.DataType): The type that should be a superset.

    Returns:
        bool: True if type_a is a subset of type_b, False otherwise.
    """
    if pa.types.is_struct(type_a) and pa.types.is_struct(type_b):
        # get the fields of B
        fields_B = {field.name: field.type for field in map(type_b.field, range(type_b.num_fields))}

        # test against the fields of A
        return all(
            ((field.name in fields_B) and _is_type_subset(field.type, fields_B[field.name]))
            for field in map(type_a.field, range(type_a.num_fields))
        )

    elif pa.types.is_list(type_a) and pa.types.is_list(type_b):
        # For lists, the value type should also be a subset
        return _is_type_subset(type_a.value_type, type_b.value_type)

    # For other types, directly compare
    return type_a.equals(type_b)


T = TypeVar("T", bound=_Feature)


class FeatureFactoryFromArrowType(FeatureFactoryFromInstance[T]):
    def __init__(self, pa_type: ArrowType) -> None:
        inst = self._create_instance(pa_type, FeatureKey(), "DummyNodeId", _DummyDataFlowGraph())
        # initialize feature factory
        super(FeatureFactoryFromArrowType, self).__init__(inst, strict=False)

    def _create_instance(
        self, pa_type: ArrowType, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph
    ) -> _Feature:
        if pa.types.is_struct(pa_type):
            mapping_type = _build_feature_type_from_arrow_type(pa_type)
            fields = map(pa_type.field, range(pa_type.num_fields))

            return mapping_type(
                _key=_key,
                _node_id=_node_id,
                _graph=_graph,
                _factories={
                    field.name: FeatureFactoryFromArrowType(field.type) for field in fields
                },
            )

        elif pa.types.is_list(pa_type):
            arrow_item_type = pa_type.value_type
            item_type = _build_feature_type_from_arrow_type(arrow_item_type)
            return Sequence[item_type](
                _key=_key,
                _node_id=_node_id,
                _graph=_graph,
                _itemtype=item_type,
                _factory=FeatureFactoryFromArrowType(arrow_item_type),
                _length=-1,
            )

        if isinstance(pa_type, ArrowType):
            scalar = ARROW_SCALAR_TYPE_TO_FEATURE_MAPPING[str(pa_type)]
            return scalar(_key, _node_id, _graph)

        raise TypeError()


class FeatureFactoryFromHuggingFace(FeatureFactoryFromArrowType[T]):
    def __init__(self, feature: FeatureType) -> None:
        if isinstance(feature, datasets.Features):
            # convert hf features to arrow feature
            pa_type = pa.struct(feature.arrow_schema)

        else:
            # build a features instance and get the arrow type of the field
            feature = datasets.Features({"field": feature})
            pa_type = feature.arrow_schema.field("field").type

        super(FeatureFactoryFromHuggingFace, self).__init__(pa_type)


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
            if isinstance(self.array.type, pa.FixedShapeTensorType):
                raise NotImplementedError()

            if isinstance(self.array.type, pa.ListArray):
                array = pa.compute.list_element(self.array, key)
                return ArrowIndexingWrapper(array)

        elif isinstance(key, slice):
            # cannot overwrite buckets
            assert self._buckets is None

            if isinstance(self.array.type, pa.FixedShapeTensorType):
                raise NotImplementedError()

            elif isinstance(self.array.type, pa.ListType):
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
