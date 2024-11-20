"""Type System Module.

This module defines a comprehensive type system for representing data types used in a structured
framework. It includes abstractions for primitive types as well as nested structures including
sequence and mapping types.
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pydantic
from pydantic_core import core_schema


class Type(ABC):
    """Abstract base class for types in the system."""

    @property
    @abstractmethod
    def arrow_type(self) -> pa.DataType:
        """Returns the corresponding :code:`PyArrow` data type.

        Returns:
            pa.DataType: The :code:`PyArrow` data type representation.
        """
        ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Generates and returns the pydantic core schema.

        Args:
            cls (type): The class for which to generate the schema.
            source_type (Any): The source type that is being validated.
            handler (pydantic.GetCoreSchemaHandler): A handler function used to generate
                the schema for the class.

        Returns:
            core_schema.CoreSchema: The generated core schema for the :class:`Feature` class,
                indicating that it is an instance schema.
        """
        return core_schema.is_instance_schema(cls)


@dataclass(eq=True, frozen=True)
class PrimitiveType(Type):
    """Represents a primitive type in the typing system."""

    _arrow_type: pa.DataType
    """The underlying :code:`PyArrow` data type."""

    @property
    def arrow_type(self) -> pa.DataType:
        """Returns the corresponding :code:`PyArrow` data type for this primitive type.

        Returns:
            pa.DataType: The :code:`PyArrow` data type representation.
        """
        return self._arrow_type


# Primitive Types
BoolType = PrimitiveType(pa.bool_())
"""Boolean type."""

StringType = PrimitiveType(pa.string())
"""String type."""

# Signed Integer Types
Int8Type = PrimitiveType(pa.int8())
"""8-bit signed integer type."""

Int16Type = PrimitiveType(pa.int16())
"""16-bit signed integer type."""

Int32Type = PrimitiveType(pa.int32())
"""32-bit signed integer type."""

Int64Type = PrimitiveType(pa.int64())
"""64-bit signed integer type."""

# Unsigned Integer Types
UInt8Type = PrimitiveType(pa.uint8())
"""An 8-bit unsigned integer type."""

UInt16Type = PrimitiveType(pa.uint16())
"""16-bit unsigned integer type."""

UInt32Type = PrimitiveType(pa.uint32())
"""32-bit unsigned integer type."""

UInt64Type = PrimitiveType(pa.uint64())
"""64-bit unsigned integer type."""

# Floating Point Types
Float16Type = PrimitiveType(pa.float16())
"""16-bit floating point type."""

Float32Type = PrimitiveType(pa.float32())
"""32-bit floating point type."""

Float64Type = PrimitiveType(pa.float64())
"""64-bit floating point type."""

# Constant
UNDEFINED_SEQUENCE_LENGTH = 2**32 - 1
"""A constant representing an undefined sequence length.

This value is used when the length of a sequence is unknown or unspecified.
"""


@dataclass(eq=True, frozen=True)
class SequenceType(Type, typing.Sequence):
    """Represents a sequence type in the typing system."""

    value_type: Type
    """The type of elements within the sequence."""

    length: int = UNDEFINED_SEQUENCE_LENGTH
    """The length of the sequence

    Set to :code:`UNDEFINED_SEQUENCE_LENGTH` in case of dynamic sequence lengths.
    """

    @property
    def arrow_type(self) -> pa.ListType:
        """Returns the corresponding :code:`PyArrow` data type for this sequence.

        Returns:
            pa.ListType: The :code:`PyArrow` list type representation.
        """
        length = -1 if self.length == UNDEFINED_SEQUENCE_LENGTH else self.length
        return pa.list_(self.value_type.arrow_type, list_size=length)

    def _slice_length(self, index: slice) -> int:
        """Calculates the length of the sequence resulting from slicing.

        Args:
            index (slice): The slice to apply to the sequence.

        Returns:
            int: The length of the sliced sequence, or :code:`UNDEFINED_SEQUENCE_LENGTH`
            if the original sequence length is unknown.
        """
        if self.length == UNDEFINED_SEQUENCE_LENGTH:
            return UNDEFINED_SEQUENCE_LENGTH

        # Compute the start, stop, and step of the slice based on the given length
        start, stop, step = index.indices(self.length)

        # If the step is positive, ensure stop is greater than start
        # If the step is negative, ensure stop is less than start
        if (step > 0 and start >= stop) or (step < 0 and start <= stop):
            return 0

        # Compute the length of the sliced sequence
        return max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)

    def __len__(self) -> int:
        """Returns the length of the sequence.

        Returns:
            int: The length of the sequence, or :code:`UNDEFINED_SEQUENCE_LENGTH` if unknown.
        """
        return self.length

    @typing.overload
    def __getitem__(self, index: int) -> Type:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> SequenceType:
        ...

    def __getitem__(self, index: int | slice) -> Type:
        """Accesses an element or a subsequence of the sequence.

        Args:
            index (int | slice): The index or slice to access.

        Returns:
            Type | SequenceType: The element type at the given index, or a new
            :class:`SequenceType` representing the subsequence.

        Raises:
            IndexError: If the index is out of bounds for the sequence.
        """
        if isinstance(index, int):
            if index >= self.length:
                raise IndexError(
                    f"Index {index} out of bounds for sequence of length {self.length}"
                )

            return self.value_type

        assert isinstance(index, slice)
        return SequenceType(self.value_type, self._slice_length(index))

    def __eq__(self, other: SequenceType) -> bool:
        """Checks equality between this sequence and another.

        Args:
            other (SequenceType): The other sequence to compare.

        Returns:
            bool: Whether the two sequences are considered equal.
        """
        if not isinstance(other, SequenceType):
            return False

        # TODO: decide if unkown == known length is ok
        return self.value_type == other.value_type and (self.length == other.length)


@dataclass(eq=True, frozen=True)
class MappingType(Type, typing.Mapping):
    """Represents a mapping type in the typing system.

    This type associates field names with their corresponding types.
    """

    fields: tuple[tuple[str, Type]]
    """The fields in the mapping, represented as a tuple of key-type pairs."""

    @property
    def arrow_type(self) -> pa.StructType:
        """Returns the corresponding :code:`PyArrow` data type for this mapping.

        Returns:
            pa.StructType: The :code:`PyArrow` struct type representation.
        """
        return pa.struct([(key, field.arrow_type) for key, field in self.fields])

    @property
    def arrow_schema(self) -> pa.Schema:
        """Returns the :code:`PyArrow` schema for this mapping.

        Returns:
            pa.Schema: The :code:`PyArrow` schema representation.
        """
        return pa.schema(self.arrow_type)

    def __len__(self) -> int:
        """Returns the number of fields in the mapping.

        Returns:
            int: The number of fields.
        """
        return len(self.fields)

    def __iter__(self) -> typing.Iterable[str]:
        """Iterates over the keys in the mapping.

        Returns:
            Iterable[str]: An iterator over the keys.
        """
        return (key for key, _ in self.fields)

    def __getitem__(self, key: str) -> Type:
        """Gets the type associated with a key in the mapping.

        Args:
            key (str): The key to look up.

        Returns:
            Type: The type associated with the key.

        Raises:
            KeyError: If the key is not present in the mapping.
        """
        return dict(self.fields)[key]

    def __eq__(self, other: MappingType) -> bool:
        """Checks equality between this mapping and another.

        Args:
            other (MappingType): The other mapping to compare.

        Returns:
            bool: Whether the two mappings are considered equal.
        """
        if not isinstance(other, MappingType):
            return False

        # order of fields doesn't matter
        return dict(self.fields) == dict(other.fields)

    @classmethod
    def from_dict(cls, fields: dict[str, Type]) -> MappingType:
        """Creates a :class:`MappingType` instance from a dictionary of fields.

        Args:
            fields (dict[str, Type]): A dictionary mapping keys to their types.

        Returns:
            MappingType: A new :class:`MappingType` instance.
        """
        return cls(fields=tuple((key, fields[key]) for key in sorted(fields.keys())))
