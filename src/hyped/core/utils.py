from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TypeVar, get_args

import datasets
import numpy as np
import numpy.typing
import pyarrow as pa
import pydantic
from datasets.features.features import FeatureType

from hyped.common.typing import ArrowType, NodeId

from .abstract import AbstractDataFlowGraph
from .features.factories import BaseFeatureFactory
from .features.feature_key import FeatureKey
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

HF_VALUE_DTYPE_TO_FEATURE_MAPPING: dict[str, _Primitive] = {
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
    "float16": _Float16,
    "float32": _Float32,
    "float64": _Float64,
}

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


T = TypeVar("T", bound=_Feature)


@dataclass
class FeatureFactoryFromArrowType(BaseFeatureFactory[T]):
    pa_type: ArrowType

    @property
    def _pa_type(self) -> ArrowType:
        return self.pa_type

    def _create_instance(
        self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph
    ) -> Any:
        if isinstance(self.pa_type, pa.StructType):
            mapping_type = _build_feature_type_from_arrow_type(self.pa_type)
            fields = map(self.pa_type.field, range(self.pa_type.num_fields))

            return mapping_type(
                _key=_key,
                _node_id=_node_id,
                _graph=_graph,
                _factories={
                    field.name: FeatureFactoryFromArrowType(field.type) for field in fields
                },
            )

        if isinstance(self.pa_type, pa.ListType):
            arrow_item_type = self.pa_type.value_type
            item_type = _build_feature_type_from_arrow_type(arrow_item_type)
            return Sequence[item_type](
                _key=_key,
                _node_id=_node_id,
                _graph=_graph,
                _itemtype=item_type,
                _factory=FeatureFactoryFromArrowType(arrow_item_type),
                _length=-1,
            )

        if isinstance(self.pa_type, ArrowType):
            scalar = ARROW_SCALAR_TYPE_TO_FEATURE_MAPPING[str(self.pa_type)]
            return scalar(_key, _node_id, _graph)

        raise TypeError()

    def __call__(self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph) -> T:
        # get the type hint if it is provided
        hint = getattr(self, "__orig_class__", None)
        hint = get_args(hint) if hint is not None else []
        hint = None if len(hint) == 0 else hint[0]
        assert (hint is None) or (not isinstance(hint, TypeVar))

        # create the instance
        inst = self._create_instance(_key, _node_id, _graph)

        if hint is None:
            # no validation
            return inst

        try:
            # validate and convert the type instance
            adapter = pydantic.TypeAdapter(hint)
            return adapter.validate_python(inst, context={"strict": True})

        except pydantic.ValidationError:
            raise RuntimeError()


def _build_arrow_type_from_hf_feature(feature: FeatureType) -> ArrowType:
    if isinstance(feature, datasets.Features):
        field_names = sorted(feature.keys())
        fields = [(n, _build_arrow_type_from_hf_feature(feature[n])) for n in field_names]
        return pa.struct(fields)

    if isinstance(feature, datasets.Sequence):
        if feature.length >= 0:
            raise NotImplementedError()

        else:
            return pa.list_(_build_arrow_type_from_hf_feature(feature.feature))

    if isinstance(feature, datasets.Value):
        return HF_VALUE_DTYPE_TO_FEATURE_MAPPING[feature.dtype]._primitive_pa_type


class FeatureFactoryFromHuggingFace(FeatureFactoryFromArrowType[T]):
    def __init__(self, feature: FeatureType) -> None:
        # convert hf features to arrow feature
        pa_type = _build_arrow_type_from_hf_feature(feature)
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
