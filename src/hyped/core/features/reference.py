"""This module provides the classes for referencing (sub-)features within a data flow system.

The key classes in this module are:
1. **FeatureKey**: A class representing a key used to index features and examples. It allows
    flexible indexing with various types such as strings, integers, and slices. It also supports
    advanced operations like slicing and hashing, making it essential for navigating and accessing
    features in a computational graph.
2. **Reference**: A class that encapsulates a feature key, node identifier, and a reference to the
    data flow graph. This class is used to track specific features in the context of a data
    processing or computational graph, facilitating the connection between features and their
    corresponding nodes and graphs.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from ..abstract import AbstractDataFlowGraph
from .types import MappingType, SequenceType, Type

NodeId: typing.TypeAlias = str
"""Node ID type in the data flow graph.

Represents the identifier for a node within a data flow graph. This is typically a string that
uniquely identifies a node, allowing for the tracking and referencing of nodes within the graph
structure.
"""


class FeatureKey(tuple[int | str | slice]):
    """Feature Key used to index specific sub-features.

    It represents the indexing operations that need to be applied to the
    feature to get a referenced sub-feature.

    Arguments:
        *key (str | int | slice): Key entries.
    """

    __slots__ = ()

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

    def index_array(self, array: pa.Array) -> pa.Array:
        """Index into the given :code:`PyArrow` array using the feature key.

        This method applies each key entry in the :class:`FeatureKey` to the corresponding level
        of the Arrow array, and returns the resulting indexed array. The indexing is performed
        according to the type of the key entry, which can be a string (for struct fields), an
        integer (for list elements), or a slice (for list slicing).

        Arguments:
            array (pa.Array): The :code:`PyArrow` array to index.

        Returns:
            pa.Array: The indexed :code:`PyArrow` array after applying the feature key.

        Raises:
            TypeError: If there is a mismatch between the key entry type and the array type.
            RuntimeError: If unsupported slicing is attempted on lists of unknown length.
        """
        for i, key_entry in enumerate(self):
            if isinstance(key_entry, str) and pa.types.is_struct(array.type):
                array = pa.compute.struct_field(array, key_entry)

            elif isinstance(key_entry, int) and (
                pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type)
            ):
                if key_entry < 0:
                    # check if length of list is fixed
                    if not pa.types.is_fixed_size_list(array.type):
                        raise IndexError(
                            "Unsupported indexing attempted on lists of unknown length. "
                            "Negative indices are not allowed for lists with unknown lengths, "
                            f"got index {key_entry}."
                        )

                    # update key entry
                    key_entry = array.type.list_size + key_entry

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

                    if (start < 0) or ((stop is not None) and (stop < 0)):
                        # not supported for lists of unkown length
                        raise IndexError(
                            "Unsupported slicing attempted on lists of unknown length. "
                            "Negative indices or stop values are not allowed for lists with "
                            f"unknown lengths, got slice ({start}, {stop}, {step})."
                        )

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

                    # unflatten the array using the nested ids
                    array = [
                        pa.compute.take(flat_array, idx, boundscheck=False) for idx in nested_ids
                    ]
                    array = pa.array(
                        array,
                        type=pa.list_(
                            flat_array.type,
                            array.type.list_size
                            if pa.types.is_fixed_size_list(flat_array.type)
                            else -1,
                        ),
                    )

                return array

            else:
                raise TypeError(
                    f"Cannot apply key entry '{key_entry}' to array of type '{array.type}'"
                )

        return array

    def index_dtype(self, dtype: Type) -> Type:
        """Index into the given data type using the feature key.

        This method applies each key entry in the :class:`FeatureKey` to the corresponding level
        of the given data type. It returns the resulting data type after the indexing. The
        indexing is performed according to the type of the key entry, which can be an integer or
        string (for :class:`SequenceType` or :class:`MappingType`), or a slice
        (for :class:`SequenceType`).

        Arguments:
            dtype (Type): The data type to index.

        Returns:
            Type: The resulting data type after applying the feature key.

        Raises:
            TypeError: If there is a mismatch between the key entry type and the data type.
        """
        for i, key_entry in enumerate(self):
            if (isinstance(key_entry, int) and isinstance(dtype, SequenceType)) or (
                isinstance(key_entry, str) and isinstance(dtype, MappingType)
            ):
                dtype = dtype[key_entry]

            elif isinstance(key_entry, slice) and isinstance(dtype, SequenceType):
                return SequenceType(
                    value_type=self[i + 1 :].index_dtype(dtype.value_type),
                    length=dtype[key_entry].length,
                )

            else:
                raise TypeError(f"Cannot apply key entry '{key_entry}' to data type '{dtype}'")

        return dtype


class DummyDataFlowGraph(AbstractDataFlowGraph):
    """Dummy Data Flow Graph."""

    def __init__(self) -> None:
        """Initialize Dummy Data Flow Graph."""
        pass


@dataclass(eq=True, frozen=True)
class Reference:
    """Represents a reference to a specific feature, node, and graph in a data flow system.

    This class encapsulates a feature key, a node identifier, and a reference to a data flow graph.
    It is used to track and reference elements in the context of a larger data processing or
    computational graph.
    """

    _key: FeatureKey = FeatureKey()
    """The key identifying the feature being referenced."""

    _node_id: NodeId = "DummyNodeId"
    """The unique identifier for the node within the graph."""

    _graph: AbstractDataFlowGraph = DummyDataFlowGraph()
    """The data flow graph that contains the node."""
