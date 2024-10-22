"""Provides the FeatureKey class.

This module defines the FeatureKey class, which is used to index features and examples
within a dataset. The FeatureKey class supports indexing with strings, integers, and
slices, and integrates with pydantic for schema validation.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from .typing import ArrowType, Batch, FeatureKeyAlias


class FeatureKey(FeatureKeyAlias):
    """Feature Key used to index features and examples.

    Arguments:
        *key (str | int | slice): Key entries.
    """

    @classmethod
    def from_tuple(cls, key: FeatureKeyAlias) -> FeatureKey:
        """Generate a FeatureKey from a tuple.

        Arguments:
            key (FeatureKeyAlias): Key entries.

        Returns:
            FeatureKey: Generated feature key.
        """
        return cls(*key)

    def __new__(cls, *key: str | int | slice) -> None:
        """Instantiate a new FeatureKey.

        Arguments:
            *key (str | int | slice): Key entries.
        """
        if len(key) == 1 and isinstance(key[0], tuple):
            return FeatureKey.from_tuple(key[0])

        if len(key) > 0 and not isinstance(key[0], str):
            raise ValueError(
                "First entry of a feature key must be a string, got %s." % repr(key[0])
            )

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
        if isinstance(idx, slice) and ((idx.start == 0) or (idx.start is None)):
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

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Integrate feature key with pydantic.

        Arguments:
            source_type (Any): Source type for the schema.
            handler (GetCoreSchemaHandler): Handler for the core schema.

        Returns:
            CoreSchema: The integrated pydantic core schema.
        """
        return core_schema.no_info_after_validator_function(cls, handler(str | tuple))

    def index_type(self, pa_type: ArrowType) -> ArrowType:
        for i, key in enumerate(self):
            if isinstance(key, str):
                # make sure the arrow type is a struct
                if not isinstance(pa_type, pa.StructType):
                    raise TypeError()
                # get the type of the field referenced by the key
                pa_type = pa_type.field(key).type

            elif isinstance(key, int):
                if isinstance(pa_type, pa.FixedShapeTensorType):
                    # make sure the type is a one-dimensional list
                    if len(pa_type.shape) != 1:
                        raise TypeError()
                    # we don't do any boundaty checks here
                    # boundary checks are handled by the feature
                    pa_type = pa_type.value_type

                elif isinstance(pa_type, pa.ListType):
                    # get the value type of the list
                    pa_type = pa_type.value_type

                else:
                    raise TypeError()

            elif isinstance(key, slice):
                sub_key = tuple.__new__(FeatureKey, self[i + 1 :])

                if isinstance(pa_type, pa.FixedShapeTensorType):
                    if len(pa_type.shape) != 1:
                        raise TypeError()

                    (length,) = pa_type.shape

                    start, stop, step = key.indices(length)
                    new_length = (stop - start) // step

                    sub_key = tuple.__new__(FeatureKey, self[i + 1 :])
                    value_type = sub_key.index_type(pa_type.value_type)

                    return pa.fixed_shape_tensor(value_type, [new_length])

                elif isinstance(pa_type, pa.ListType):
                    sub_key = tuple.__new__(FeatureKey, self[i + 1 :])
                    return pa.list_(sub_key.index_type(pa_type.value_type))

        return pa_type

    def index_batch(self, batch: Batch) -> pa.Array:
        """Index a batch of examples with the given key and retrieve the batch of values.

        Arguments:
            batch (Table): Batch of examples to index.

        Returns:
            Array: The batch of values of the examples at the given key.
        """
        gathered_batch = batch.to_struct_array()
        for key_entry in self:
            if isinstance(key_entry, str):
                gathered_batch = pa.compute.struct_field(gathered_batch, key_entry)

            elif isinstance(key_entry, int):
                gathered_batch = pa.compute.list_element(gathered_batch, key_entry)

            elif isinstance(key_entry, slice):
                start = key_entry.start if key_entry.start is not None else 0
                step = key_entry.step if key_entry.step is not None else 1
                gathered_batch = pa.compute.list_slice(gathered_batch, start, key_entry.stop, step)

        return gathered_batch

    def __hash__(self) -> int:
        """Compute the hash value of the feature key.

        Returns:
            int: The hash value of the feature key.
        """
        return hash(tuple((k.start, k.stop, k.step) if isinstance(k, slice) else k for k in self))
