"""Core typing module.

This module defines the type aliases that need to me used to define
node interfaces.
"""
from typing import Annotated, Any, TypeAlias, Union

import pyarrow as pa

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
    _String,
    _UInt8,
    _UInt16,
    _UInt32,
    _UInt64,
)

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
]


Feature: TypeAlias = Union[Any, list[Any], pa.Scalar, pa.Array, _Feature]
"""
Feature: Type alias for a feature.

Supported types include :class:`Any`, :class:`list[Any]`, :class:`pyarrow.Scalar`,
:class:`pyarrow.Array` and :class:`_Feature`.
"""

String: TypeAlias = Union[str, list[str], pa.StringScalar, pa.StringArray, _String]
"""
String: Type alias for a string.

Supported types include :class:`str`, :class:`list[str]`, :class:`pyarrow.StringScalar`,
:class:`pyarrow.StringArray` and :class:`_String`.
"""

Bool: TypeAlias = Union[bool, list[bool], pa.BooleanScalar, pa.BooleanArray, _Bool]
"""
Bool: Type alias for a boolean.

Supported types include :class:`bool`, :class:`list[bool]`, :class:`pyarrow.BooleanScalar`
:class:`pyarrow.BooleanArray` and :class:`_Bool`.
"""

UInt8: TypeAlias = Union[int, list[int], pa.UInt8Scalar, pa.UInt8Array, _UInt8]
"""
UInt8: Type alias for an 8-bit unsigned integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.UInt8Scalar`,
:class:`pyarrow.UInt8Array`, and :class:`_UInt8`.
"""

UInt16: TypeAlias = Union[int, list[int], pa.UInt16Scalar, pa.UInt16Array, _UInt16]
"""
UInt16: Type alias for a 16-bit unsigned integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.UInt16Scalar`,
:class:`pyarrow.UInt16Array`, and :class:`_UInt16`.
"""

UInt32: TypeAlias = Union[int, list[int], pa.UInt32Scalar, pa.UInt32Array, _UInt32]
"""
UInt32: Type alias for a 32-bit unsigned integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.UInt32Scalar`,
:class:`pyarrow.UInt32Array`, and :class:`_UInt32`.
"""

UInt64: TypeAlias = Union[int, list[int], pa.UInt64Scalar, pa.UInt64Array, _UInt64]
"""
UInt64: Type alias for a 64-bit unsigned integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.UInt64Scalar`,
:class:`pyarrow.UInt64Array`, and :class:`_UInt64`.
"""

UInt: TypeAlias = Union[
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
    _UInt8,
    _UInt16,
    _UInt32,
    _UInt64,
]
"""
UInt: Type alias for an unsigned integer of varying bit length.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.UInt8Scalar`,
:class:`pyarrow.UInt16Scalar`, :class:`pyarrow.UInt32Scalar`, :class:`pyarrow.UInt64Scalar`,
:class:`pyarrow.UInt8Array`, :class:`pyarrow.UInt16Array`, :class:`pyarrow.UInt32Array`,
:class:`pyarrow.UInt64Array`, :class:`_UInt8`, :class:`_UInt16`, :class:`_UInt32`, and
:class:`_UInt64`.
"""

Int8: TypeAlias = Union[int, list[int], pa.Int8Scalar, pa.Int8Array, _Int8]
"""
Int8: Type alias for an 8-bit signed integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.Int8Scalar`,
:class:`pyarrow.Int8Array`, and :class:`_Int8`.
"""

Int16: TypeAlias = Union[int, list[int], pa.Int16Scalar, pa.Int16Array, _Int16]
"""
Int16: Type alias for a 16-bit signed integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.Int16Scalar`,
:class:`pyarrow.Int16Array`, and :class:`_Int16`.
"""

Int32: TypeAlias = Union[int, list[int], pa.Int32Scalar, pa.Int32Array, _Int32]
"""
Int32: Type alias for a 32-bit signed integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.Int32Scalar`,
:class:`pyarrow.Int32Array`, and :class:`_Int32`.
"""

Int64: TypeAlias = Union[int, list[int], pa.Int64Scalar, pa.Int64Array, _Int64]
"""
Int64: Type alias for a 64-bit signed integer.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.Int64Scalar`,
:class:`pyarrow.Int64Array`, and :class:`_Int64`.
"""

Int: TypeAlias = Union[
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
    _Int8,
    _Int16,
    _Int32,
    _Int64,
]
"""
Int: Type alias for a signed integer of varying bit length.

Supported types include :class:`int`, :class:`list[int]`, :class:`pyarrow.Int8Scalar`,
:class:`pyarrow.Int16Scalar`, :class:`pyarrow.Int32Scalar`, :class:`pyarrow.Int64Scalar`,
:class:`pyarrow.Int8Array`, :class:`pyarrow.Int16Array`, :class:`pyarrow.Int32Array`,
:class:`pyarrow.Int64Array`, :class:`_Int8`, :class:`_Int16`, :class:`_Int32`, and
:class:`_Int64`.
"""

Float16: TypeAlias = Union[float, list[float], pa.HalfFloatScalar, pa.HalfFloatArray, _Float16]
"""
Float16: Type alias for a 16-bit floating-point number.

Supported types include :class:`float`, :class:`list[float]`, :class:`pyarrow.HalfFloatScalar`,
:class:`pyarrow.HalfFloatArray`, and :class:`_Float16`.
"""

Float32: TypeAlias = Union[float, list[float], pa.FloatScalar, pa.FloatArray, _Float32]
"""
Float32: Type alias for a 32-bit floating-point number.

Supported types include :class:`float`, :class:`list[float]`, :class:`pyarrow.FloatScalar`,
:class:`pyarrow.FloatArray`, and :class:`_Float32`.
"""

Float64: TypeAlias = Union[float, list[float], pa.DoubleScalar, pa.DoubleArray, _Float64]
"""
Float64: Type alias for a 64-bit floating-point number.

Supported types include :class:`float`, :class:`list[float]`, :class:`pyarrow.DoubleScalar`,
:class:`pyarrow.DoubleArray`, and :class:`_Float64`.
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

Supported types include :class:`float`, :class:`list[float]`, :class:`pyarrow.HalfFloatScalar`,
:class:`pyarrow.FloatScalar`, :class:`pyarrow.DoubleScalar`, :class:`pyarrow.HalfFloatArray`,
:class:`pyarrow.FloatArray`, :class:`pyarrow.DoubleArray`, :class:`_Float16`, :class:`_Float32`,
and :class:`_Float64`.
"""
