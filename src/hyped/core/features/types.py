from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pyarrow as pa
from pydantic_core import core_schema


class Type(ABC):
    @property
    @abstractmethod
    def arrow_type(self) -> pa.DataType:
        ...

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.is_instance_schema(cls)


@dataclass(eq=True, frozen=True)
class PrimitiveType(Type):
    _arrow_type: pa.DataType

    @property
    def arrow_type(self) -> pa.DataType:
        return self._arrow_type


BoolType = PrimitiveType(pa.bool_())
StringType = PrimitiveType(pa.string())

Int8Type = PrimitiveType(pa.int8())
Int16Type = PrimitiveType(pa.int16())
Int32Type = PrimitiveType(pa.int32())
Int64Type = PrimitiveType(pa.int64())

UInt8Type = PrimitiveType(pa.uint8())
UInt16Type = PrimitiveType(pa.uint16())
UInt32Type = PrimitiveType(pa.uint32())
UInt64Type = PrimitiveType(pa.uint64())

Float16Type = PrimitiveType(pa.float16())
Float32Type = PrimitiveType(pa.float32())
Float64Type = PrimitiveType(pa.float64())

UNDEFINED_SEQUENCE_LENGTH = 2**32 - 1


@dataclass(eq=True, frozen=True)
class SequenceType(Type, typing.Sequence):
    value_type: Type
    length: int = UNDEFINED_SEQUENCE_LENGTH

    @property
    def arrow_type(self) -> pa.DataType:
        length = -1 if self.length == UNDEFINED_SEQUENCE_LENGTH else self.length
        return pa.list_(self.value_type.arrow_type, list_size=length)

    def _slice_length(self, index: slice) -> int:
        if self.length == UNDEFINED_SEQUENCE_LENGTH:
            return UNDEFINED_SEQUENCE_LENGTH  # Length of the sequence is unknown, so return -1

        # Compute the start, stop, and step of the slice based on the given length
        start, stop, step = index.indices(self.length)

        # If the step is positive, ensure stop is greater than start
        # If the step is negative, ensure stop is less than start
        if (step > 0 and start >= stop) or (step < 0 and start <= stop):
            return 0

        # Compute the length of the sliced sequence
        return max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)

    def __len__(self) -> int:
        return self.length

    @typing.overload
    def __getitem__(self, index: int) -> Type:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> SequenceType:
        ...

    def __getitem__(self, index: int | slice) -> Type:
        if isinstance(index, int):
            if index >= self.length:
                raise IndexError(
                    f"Index {index} out of bounds for sequence of length {self.length}"
                )

            return self.value_type

        assert isinstance(index, slice)
        return SequenceType(self.value_type, self._slice_length(index))

    def __eq__(self, other: SequenceType) -> bool:
        if not isinstance(other, SequenceType):
            return False

        # TODO: decide if unkown == known length is ok
        return self.value_type == other.value_type and (self.length == other.length)


@dataclass(eq=True, frozen=True)
class MappingType(Type, typing.Mapping):
    fields: tuple[tuple[str, Type]]

    @property
    def arrow_type(self) -> pa.DataType:
        return pa.struct([(key, field.arrow_type) for key, field in self.fields])

    @property
    def arrow_schema(self) -> pa.Schema:
        return pa.schema(self.arrow_type)

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> typing.Iterable[str]:
        return (key for key, _ in self.fields)

    def __getitem__(self, key: str) -> Type:
        return dict(self.fields)[key]

    @classmethod
    def from_dict(cls, fields: dict[str, Type]) -> MappingType:
        return cls(fields=tuple((key, fields[key]) for key in sorted(fields.keys())))

    def __eq__(self, other: MappingType) -> bool:
        if not isinstance(other, MappingType):
            return False

        # order of fields doesn't matter
        return dict(self.fields) == dict(other.fields)
