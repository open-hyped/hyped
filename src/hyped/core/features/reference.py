from __future__ import annotations

import typing
from dataclasses import dataclass

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from hyped.common.typing import NodeId

from ..abstract import AbstractDataFlowGraph


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

    def __getitem__(self, idx) -> FeatureKey | str | int | slice:
        """Get specific key entries of the feature key.

        Arguments:
            idx: Index to retrieve from the feature key.

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

    def index_object(self, obj: typing.Any | typing.Mapping | typing.Sequence) -> object:
        for i, key_entry in enumerate(self):
            if isinstance(key_entry, (int, str)):
                # get item from object
                obj = obj[key_entry]

            if isinstance(key_entry, slice):
                # apply the slicing operation on the object
                items = obj[key_entry]
                # build the subkey of remainding entries that needs to be applied
                # to all entries included in the slice
                next_key = tuple.__new__(FeatureKey, self[i + 1 :])
                # apply the key to all entries in the slice
                items = map(next_key.index_object, items)
                # pack processed items in object
                constructor = getattr(obj, "__slice_constructor__", type(obj))
                return constructor(items)

        return obj


@dataclass(eq=True, frozen=True)
class Reference:
    _key: FeatureKey
    _node_id: NodeId
    _graph: AbstractDataFlowGraph
