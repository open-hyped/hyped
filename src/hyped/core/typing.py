from typing import Annotated, Any, TypeAlias, Union

from .features.features import (
    Mapping,
    Sequence,
    TypeVar,
    _Bool,
    _Feature,
    _Float16,
    _Float32,
    _Float64,
    _Int8,
    _Int16,
    _Int32,
    _Int64,
    _String,
    _UInt8,
    _UInt16,
    _UInt32,
    _UInt64,
)
from .features.validators import Default, Len, TypeResolver, TypeValidator

__all__ = [
    "Union",
    "Annotated",
    "Feature",
    "String",
    "Bool",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Sequence",
    "Mapping",
    "Len",
    "TypeValidator",
    "TypeResolver",
    "TypeVar",
]


Feature: TypeAlias = Annotated[Union[Any, _Feature], Default(_Feature)]
"""
Feature: Type alias for a feature.

Supported types include :class:`Any` and :class:`_Feature`.

Defaults to custom feature type (:class:`_Feature`).
"""

String: TypeAlias = Annotated[Union[str, _String], Default(_String)]
"""
String: Type alias for a string.

Supported types include :class:`str` and :class:`_String`.

Defaults to custom string type (:class:`_String`).
"""

Bool: TypeAlias = Annotated[Union[bool, _Bool], Default(_Bool)]
"""
Bool: Type alias for a boolean.

Supported types include :class:`bool` and :class:`_Bool`.

Defaults to custom boolean type (:class:`_Bool`).
"""

UInt8: TypeAlias = Annotated[Union[int, _UInt8], Default(_UInt8)]
"""
UInt8: Type alias for an 8-bit unsigned integer.

Supported types include :class:`int` and :class:`_UInt8`.

Defaults to custom 8-bit unsigned integer type (:class:`_UInt8`).
"""

UInt16: TypeAlias = Annotated[Union[int, _UInt16], Default(_UInt16)]
"""
UInt16: Type alias for a 16-bit unsigned integer.

Supported types include :class:`int` and :class:`_UInt16`.

Defaults to custom 16-bit unsigned integer type (:class:`_UInt16`).
"""

UInt32: TypeAlias = Annotated[Union[int, _UInt32], Default(_UInt32)]
"""
UInt32: Type alias for a 32-bit unsigned integer.

Supported types include :class:`int` and :class:`_UInt32`.

Defaults to custom 32-bit unsigned integer type (:class:`_UInt32`).
"""

UInt64: TypeAlias = Annotated[Union[int, _UInt64], Default(_UInt64)]
"""
UInt64: Type alias for a 64-bit unsigned integer.

Supported types include :class:`int` and :class:`_UInt64`.

Defaults to custom 64-bit unsigned integer type (:class:`_UInt64`).
"""

UInt: TypeAlias = Annotated[Union[int, _UInt8, _UInt16, _UInt32, _UInt64], Default(_UInt64)]
"""
UInt: Type alias for an unsigned integer of varying bit length.

Supported types include :class:`int`, :class:`_UInt8`, :class:`_UInt16`, :class:`_UInt32`, and
:class:`_UInt64`.

Defaults to custom 64-bit unsigned integer type (:class:`_UInt64`).
"""

Int8: TypeAlias = Annotated[Union[int, _Int8, _UInt8], Default(_Int8)]
"""
Int8: 8-bit integer type alias.

Supported types include :class:`int`, :class:`_Int8` (signed), and :class:`_UInt8` (unsigned).

Defaults to signed 8-bit integer (:class:`_Int8`).
"""

Int16: TypeAlias = Annotated[Union[int, _Int16, _UInt16], Default(_Int16)]
"""
Int16: 16-bit integer type alias.

Supported types include :class:`int`, :class:`_Int16` (signed), and :class:`_UInt16` (unsigned).

Defaults to signed 16-bit integer (:class:`_Int16`).
"""

Int32: TypeAlias = Annotated[Union[int, _Int32, _UInt32], Default(_Int32)]
"""
Int32: 32-bit integer type alias.

Supported types include :class:`int`, :class:`_Int32` (signed), and :class:`_UInt32` (unsigned).

Defaults to signed 32-bit integer (:class:`_Int32`).
"""

Int64: TypeAlias = Annotated[Union[int, _Int64, _UInt64], Default(_Int64)]
"""
Int64: 64-bit integer type alias.

Supported types include :class:`int`, :class:`_Int64` (signed), and :class:`_UInt64` (unsigned).

Defaults to signed 64-bit integer (:class:`_Int64`).
"""

Int: TypeAlias = Annotated[
    Union[int, _Int8, _Int16, _Int32, _Int64, _UInt8, _UInt16, _UInt32, _UInt64], Default(_Int64)
]
"""
Int: Type alias for an integer of varying bit length.

Supported types include :class:`int`, :class:`_Int8`, :class:`_Int16`, :class:`_Int32`,
:class:`_Int64`, :class:`_UInt8`, :class:`_UInt16`, :class:`_UInt32`, and :class:`_UInt64`.

Defaults to signed 64-bit integer (:class:`_Int64`).
"""

Float16: TypeAlias = Annotated[Union[float, _Float16], Default(_Float16)]
"""
Float16: 16-bit floating-point type alias.

Supported types include :class:`float` and :class:`_Float16`.

Defaults to custom 16-bit floating-point type (:class:`_Float16`).
"""

Float32: TypeAlias = Annotated[Union[float, _Float32], Default(_Float32)]
"""
Float32: 32-bit floating-point type alias.

Supported types include :class:`float` and :class:`_Float32`.

Defaults to custom 32-bit floating-point type (:class:`_Float32`).
"""

Float64: TypeAlias = Annotated[Union[float, _Float64], Default(_Float64)]
"""
Float64: 64-bit floating-point type alias.

Supported types include :class:`float` and :class:`_Float64`.

Defaults to custom 64-bit floating-point type (:class:`_Float64`).
"""

Float: TypeAlias = Annotated[Union[float, _Float16, _Float32, _Float64], Default(_Float64)]
"""
Float: Type alias for a floating-point number of varying precision.

Supported types include :class:`float`, :class:`_Float16`, :class:`_Float32`, and :class:`_Float64`.

Defaults to custom 64-bit floating-point type (:class:`_Float64`).
"""
