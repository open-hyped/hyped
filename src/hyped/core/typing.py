"""Core typing module.

This module defines the type aliases that need to me used to define
node interfaces.
"""
from typing import Annotated, Any, TypeAlias, TypeVar, Union

import pyarrow as pa

from .features.features import (
    _Bool,
    _Feature,
    _Float16,
    _Float32,
    _Float64,
    _Int8,
    _Int16,
    _Int32,
    _Int64,
    _Mapping,
    _Sequence,
    _String,
    _UInt8,
    _UInt16,
    _UInt32,
    _UInt64,
)
from .features.validators import Len

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
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Sequence",
    "Mapping",
    "Len",
]

Feature: TypeAlias = Union[_Feature, Any, list[Any], pa.Scalar, pa.Array]
"""
Feature: Type alias for a feature.

Supported types include:
- Feature Types: :class:`_Feature`
- Built-in types: :class:`Any`, :class:`list[Any]`
- PyArrow scalar types: :class:`pyarrow.Scalar`
- PyArrow array types: :class:`pyarrow.Array`
"""

String: TypeAlias = Union[_String, str, list[str], pa.StringScalar, pa.StringArray]
"""
String: Type alias for a string.

Supported types include:
- Feature Types: :class:`_String`
- Built-in types: :class:`str`, :class:`list[str]`
- PyArrow scalar types: :class:`pyarrow.StringScalar`
- PyArrow array types: :class:`pyarrow.StringArray`
"""

Bool: TypeAlias = Union[_Bool, bool, list[bool], pa.BooleanScalar, pa.BooleanArray]
"""
Bool: Type alias for a boolean.

Supported types include:
- Feature Types: :class:`_Bool`
- Built-in types: :class:`bool`, :class:`list[bool]`
- PyArrow scalar types: :class:`pyarrow.BooleanScalar`
- PyArrow array types: :class:`pyarrow.BooleanArray`
"""

UInt8: TypeAlias = Union[_UInt8, int, list[int], pa.UInt8Scalar, pa.UInt8Array]
"""
UInt8: Type alias for an 8-bit unsigned integer.

Supported types include:
- Feature Types: :class:`_UInt8`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt8Scalar`
- PyArrow array types: :class:`pyarrow.UInt8Array`
"""

UInt16: TypeAlias = Union[_UInt16, int, list[int], pa.UInt16Scalar, pa.UInt16Array]
"""
UInt16: Type alias for a 16-bit unsigned integer.

Supported types include:
- Feature Types: :class:`_UInt16`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt16Scalar`
- PyArrow array types: :class:`pyarrow.UInt16Array`
"""

UInt32: TypeAlias = Union[_UInt32, int, list[int], pa.UInt32Scalar, pa.UInt32Array]
"""
UInt32: Type alias for a 32-bit unsigned integer.

Supported types include:
- Feature Types: :class:`_UInt32`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt32Scalar`
- PyArrow array types: :class:`pyarrow.UInt32Array`
"""

UInt64: TypeAlias = Union[_UInt64, int, list[int], pa.UInt64Scalar, pa.UInt64Array]
"""
UInt64: Type alias for a 64-bit unsigned integer.

Supported types include:
- Feature Types: :class:`_UInt64`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt64Scalar`
- PyArrow array types: :class:`pyarrow.UInt64Array`
"""

UInt: TypeAlias = Union[
    _UInt8,
    _UInt16,
    _UInt32,
    _UInt64,
    int,
    list[int],
    pa.UInt8Scalar,
    pa.UInt16Scalar,
    pa.UInt32Scalar,
    pa.UInt64Scalar,
    pa.UInt8Array,
    pa.UInt16Array,
    pa.UInt32Array,
    pa.UInt64Array,
]
"""
UInt: Type alias for an unsigned integer of varying bit length.

Supported types include:
- Feature Types: :class:`_UInt8`, :class:`_UInt16`, :class:`_UInt32`, :class:`_UInt64`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.UInt8Scalar`, :class:`pyarrow.UInt16Scalar`,
  :class:`pyarrow.UInt32Scalar`, :class:`pyarrow.UInt64Scalar`
- PyArrow array types: :class:`pyarrow.UInt8Array`, :class:`pyarrow.UInt16Array`,
  :class:`pyarrow.UInt32Array`, :class:`pyarrow.UInt64Array`
"""

Int8: TypeAlias = Union[_Int8, int, list[int], pa.Int8Scalar, pa.Int8Array]
"""
Int8: Type alias for an 8-bit signed integer.

Supported types include:
- Feature Types: :class:`_Int8`
- Built-in types: :class:`int`, :class:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int8Scalar`
- PyArrow array types: :class:`pyarrow.Int8Array`
"""

Int16: TypeAlias = Union[_Int16, int, list[int], pa.Int16Scalar, pa.Int16Array]
"""
Int16: Type alias for a 16-bit signed integer.

Supported types include:
- Feature Types: :class:`_Int16`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int16Scalar`
- PyArrow array types: :class:`pyarrow.Int16Array`
"""

Int32: TypeAlias = Union[_Int32, int, list[int], pa.Int32Scalar, pa.Int32Array]
"""
Int32: Type alias for a 32-bit signed integer.

Supported types include:
- Feature Types: :class:`_Int32`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int32Scalar`
- PyArrow array types: :class:`pyarrow.Int32Array`
"""

Int64: TypeAlias = Union[_Int64, int, list[int], pa.Int64Scalar, pa.Int64Array]
"""
Int64: Type alias for a 64-bit signed integer.

Supported types include:
- Feature Types: :class:`_Int64`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int64Scalar`
- PyArrow array types: :class:`pyarrow.Int64Array`
"""

Int: TypeAlias = Union[
    _Int8,
    _Int16,
    _Int32,
    _Int64,
    int,
    list[int],
    pa.Int8Scalar,
    pa.Int16Scalar,
    pa.Int32Scalar,
    pa.Int64Scalar,
    pa.Int8Array,
    pa.Int16Array,
    pa.Int32Array,
    pa.Int64Array,
]
"""
Int: Type alias for a signed integer of varying bit length.

Supported types include:
- Feature Types: :class:`_Int8`, :class:`_Int16`, :class:`_Int32`, :class:`_Int64`
- Built-in types: :class:`int`, :code:`list[int]`
- PyArrow scalar types: :class:`pyarrow.Int8Scalar`, :class:`pyarrow.Int16Scalar`,
  :class:`pyarrow.Int32Scalar`, :class:`pyarrow.Int64Scalar`
- PyArrow array types: :class:`pyarrow.Int8Array`, :class:`pyarrow.Int16Array`,
  :class:`pyarrow.Int32Array`, :class:`pyarrow.Int64Array`
"""

Float16: TypeAlias = Union[_Float16, float, list[float], pa.HalfFloatScalar, pa.HalfFloatArray]
"""
Float16: Type alias for a 16-bit floating-point number.

Supported types include:
- Feature Types: :class:`_Float16`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.HalfFloatScalar`
- PyArrow array types: :class:`pyarrow.HalfFloatArray`
"""

Float32: TypeAlias = Union[_Float32, float, list[float], pa.FloatScalar, pa.FloatArray]
"""
Float32: Type alias for a 32-bit floating-point number.

Supported types include:
- Feature Types: :class:`_Float32`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.FloatScalar`
- PyArrow array types: :class:`pyarrow.FloatArray`
"""

Float64: TypeAlias = Union[_Float64, float, list[float], pa.DoubleScalar, pa.DoubleArray]
"""
Float64: Type alias for a 64-bit floating-point number.

Supported types include:
- Feature Types: :class:`_Float64`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.DoubleScalar`
- PyArrow array types: :class:`pyarrow.DoubleArray`
"""

Float: TypeAlias = Union[
    float,
    list[float],
    pa.HalfFloatScalar,
    pa.FloatScalar,
    pa.DoubleScalar,
    pa.HalfFloatArray,
    pa.FloatArray,
    pa.DoubleArray,
    _Float16,
    _Float32,
    _Float64,
]
"""
Float: Type alias for a floating-point number of varying precision.

Supported types include:
- Feature Types: :class:`_Float16`, :class:`_Float32`, :class:`_Float64`
- Built-in types: :class:`float`, :code:`list[float]`
- PyArrow scalar types: :class:`pyarrow.HalfFloatScalar`, :class:`pyarrow.FloatScalar`,
  :class:`pyarrow.DoubleScalar`
- PyArrow array types: :class:`pyarrow.HalfFloatArray`, :class:`pyarrow.FloatArray`,
  :class:`pyarrow.DoubleArray`
"""

T = TypeVar("T")

Sequence: TypeAlias = Union[_Sequence[T], list[T], list[list[T]], pa.ListScalar, pa.ListArray]
"""
Sequence: Type alias for a sequence of items of a specified type.

Supported types include:
- Feature Types: :code:`_Sequence[T]`
- Built-in types: :code:`list[T]`, :code:`list[list[T]]`
- PyArrow scalar types: :class:`pyarrow.ListScalar`
- PyArrow array types: :class:`pyarrow.ListArray`
"""

# TODO:
Mapping: TypeAlias = _Mapping
