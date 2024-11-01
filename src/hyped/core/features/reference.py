from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from hyped.common.typing import NodeId

from ..abstract import AbstractDataFlowGraph
from .types import MappingType, SequenceType, Type


class FeatureKey(tuple[int | str | slice]):
    """Feature Key used to index features and examples.

    Arguments:
        *key (str | int | slice): Key entries.
    """

    @classmethod
    def from_tuple(cls, key: tuple[int | str | slice]) -> FeatureKey:
        """Generate a FeatureKey from a tuple.

        Arguments:
            key (FeatureKeyAlias): Key entries.

        Returns:
            FeatureKey: Generated feature key.
        """
        return cls(*key)

    def __new__(cls, *key: str | int | slice) -> FeatureKey:
        """Instantiate a new FeatureKey.

        Arguments:
            *key (str | int | slice): Key entries.
        """
        if len(key) == 1 and isinstance(key[0], tuple):
            return FeatureKey.from_tuple(key[0])

        for key_entry in key:
            if not isinstance(key_entry, (str, int, slice)):
                raise TypeError(
                    "Feature key entries must be either str, int or slice "
                    "objects, got %s" % key_entry
                )

        return tuple.__new__(cls, key)

    @typing.overload
    def __getitem__(self, idx: slice) -> FeatureKey:
        ...

    @typing.overload
    def __getitem__(self, idx: int) -> str | int | slice:
        ...

    def __getitem__(self, idx: int | slice) -> FeatureKey | str | int | slice:
        """Get specific key entries of the feature key.

        Arguments:
            idx (int | slice): Index to retrieve from the feature key.

        Returns:
            FeatureKey | str | int | slice: The retrieved key entry or a new FeatureKey.
        """
        if isinstance(idx, slice):
            return FeatureKey(*super(FeatureKey, self).__getitem__(idx))
        return super(FeatureKey, self).__getitem__(idx)

    def __str__(self) -> str:
        """String representation of the feature key.

        Returns:
            str: String representation of the feature key.
        """
        return "(%s)" % "->".join(map(str, self))

    def __repr__(self) -> str:
        """String representation of the feature key.

        Returns:
            str: String representation of the feature key.
        """
        return "FeatureKey%s" % str(self)

    def __hash__(self) -> int:
        """Compute the hash value of the feature key.

        Returns:
            int: The hash value of the feature key.
        """
        return hash(tuple((k.start, k.stop, k.step) if isinstance(k, slice) else k for k in self))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Integrate feature key with pydantic.

        Arguments:
            source_type (typing.Any): Source type for the schema.
            handler (GetCoreSchemaHandler): Handler for the core schema.

        Returns:
            CoreSchema: The integrated pydantic core schema.
        """
        return core_schema.no_info_after_validator_function(cls, handler(tuple | str))

    def index_array(self, array: pa.Array) -> pa.Array:
        for i, key_entry in enumerate(self):
            if isinstance(key_entry, str) and pa.types.is_struct(array.type):
                array = pa.compute.struct_field(array, key_entry)

            elif isinstance(key_entry, int) and (
                pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type)
            ):
                array = pa.compute.list_element(array, key_entry)

            elif isinstance(key_entry, slice) and (
                pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type)
            ):
                if pa.types.is_fixed_size_list(array.type):
                    start, stop, step = key_entry.indices(array.type.list_size)

                else:
                    # prepare the slice, cannot use .indices here because we don't know the length
                    stop = key_entry.stop
                    start = key_entry.start if key_entry.start is not None else 0
                    step = key_entry.step if key_entry.step is not None else 1

                    if (start < 0) or (stop < 0):
                        # not supported for lists of unkown length
                        raise RuntimeError()

                if key_entry != slice(None):
                    # get the list slice but only if there are actually values being omitted be
                    # slicing, otherwise (i.e. slice(None)) just keep the full list
                    array = pa.compute.list_slice(array, start, stop, step)

                if i + 1 < len(self):
                    # flatten the list for further processing
                    flat_array = pa.compute.list_flatten(array)

                    # get information needed to invert the flatten operation
                    parent_index = pa.compute.list_parent_indices(array).to_numpy()
                    nested_ids = [np.nonzero(parent_index == j)[0] for j in range(len(array))]

                    # apply the remainding key on the flattened array
                    flat_array = self[i + 1 :].index_array(flat_array)

                    # unflatten the array using the buckets
                    array = [
                        pa.compute.take(flat_array, idx, boundschecks=False) for idx in nested_ids
                    ]
                    array = pa.array(
                        array,
                        type=pa.list_(
                            flat_array.type,
                            array.type.list_size if pa.types.is_fixed_size_list(array.type) else -1,
                        ),
                    )

                return array

            else:
                # TODO: mismatch between key and array type
                raise TypeError(key_entry, array.type)

        return array

    def index_dtype(self, dtype: Type) -> Type:
        for i, key_entry in enumerate(self):
            if isinstance(key_entry, (int, str)) and (
                isinstance(dtype, (SequenceType, MappingType))
            ):
                dtype = dtype[key_entry]

            elif isinstance(key_entry, slice) and isinstance(dtype, SequenceType):
                return SequenceType(
                    value_type=self[i + 1 :].index_dtype(dtype.value_type),
                    length=dtype[key_entry].length,
                )

            else:
                # TODO: mismatch between key and array type
                raise TypeError(key_entry, dtype)

        return dtype


class DummyDataFlowGraph(AbstractDataFlowGraph):
    def __init__(self) -> None:
        pass


@dataclass(eq=True, frozen=True)
class Reference:
    _key: FeatureKey = FeatureKey()
    _node_id: NodeId = "DummyNodeId"
    _graph: AbstractDataFlowGraph = DummyDataFlowGraph()
