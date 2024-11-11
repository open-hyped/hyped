"""Core typing module.

This module defines the type aliases that need to me used to define
node interfaces.
"""
from typing import Annotated, Any, TypeAlias, Union

import pyarrow as pa

from hyped.common._pydantic import CustomType

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


Feature: TypeAlias = Union[Any, CustomType[pa.Scalar], _Feature]
"""
Feature: Type alias for a feature.

Supported types include :class:`Any`, :class:`pyarrow.Scalar` and :class:`_Feature`.
"""

String: TypeAlias = Union[str, CustomType[pa.StringScalar], _String]
"""
String: Type alias for a string.

Supported types include :class:`str`, :class:`pyarrow.StringScalar` and :class:`_String`.
"""

Bool: TypeAlias = Union[bool, CustomType[pa.BooleanScalar], _Bool]
"""
Bool: Type alias for a boolean.

Supported types include :class:`bool`, :class:`pyarrow.BooleanScalar` and :class:`_Bool`.
"""

UInt8: TypeAlias = Union[int, CustomType[pa.UInt8Scalar], _UInt8]
"""
UInt8: Type alias for an 8-bit unsigned integer.

Supported types include :class:`int`, :class:`pyarrow.UInt8Scalar` and :class:`_UInt8`.
"""

UInt16: TypeAlias = Union[int, CustomType[pa.UInt16Scalar], _UInt16]
"""
UInt16: Type alias for a 16-bit unsigned integer.

Supported types include :class:`int`, :class:`pyarrow.UInt16Scalar` and :class:`_UInt16`.
"""

UInt32: TypeAlias = Union[int, CustomType[pa.UInt32Scalar], _UInt32]
"""
UInt32: Type alias for a 32-bit unsigned integer.

Supported types include :class:`int`, :class:`pyarrow.UInt32Scalar` and :class:`_UInt32`.
"""

UInt64: TypeAlias = Union[int, CustomType[pa.UInt64Scalar], _UInt64]
"""
UInt64: Type alias for a 64-bit unsigned integer.

Supported types include :class:`int`, :class:`pyarrow.UInt64Scalar` and :class:`_UInt64`.
"""

UInt: TypeAlias = Union[
    int,
    CustomType[pa.UInt64Scalar],
    CustomType[pa.UInt32Scalar],
    CustomType[pa.UInt16Scalar],
    CustomType[pa.UInt8Scalar],
    _UInt64,
    _UInt32,
    _UInt16,
    _UInt8,
]
"""
UInt: Type alias for an unsigned integer of varying bit length.

Supported types include :class:`int`, :class:`pyarrow.UInt8Scalar`, :class:`pyarrow.UInt16Scalar`,
:class:`pyarrow.UInt32Scalar`, :class:`pyarrow.UInt64Scalar`, :class:`_UInt8`, :class:`_UInt16`,
:class:`_UInt32`, and :class:`_UInt64`.
"""

Int8: TypeAlias = Union[int, CustomType[pa.Int8Scalar], _Int8]
"""
Int8: Type alias for an 8-bit signed integer.

Supported types include :class:`int`, :class:`pyarrow.Int8Scalar`, and :class:`_Int8`.
"""

Int16: TypeAlias = Union[int, CustomType[pa.Int16Scalar], _Int16]
"""
Int16: Type alias for a 16-bit signed integer.

Supported types include :class:`int`, :class:`pyarrow.Int16Scalar`, and :class:`_Int16`.
"""

Int32: TypeAlias = Union[int, CustomType[pa.Int32Scalar], _Int32]
"""
Int32: Type alias for a 32-bit signed integer.

Supported types include :class:`int`, :class:`pyarrow.Int32Scalar`, and :class:`_Int32`.
"""

Int64: TypeAlias = Union[int, CustomType[pa.Int64Scalar], _Int64]
"""
Int64: Type alias for a 64-bit signed integer.

Supported types include :class:`int`, :class:`pyarrow.Int64Scalar`, and :class:`_Int64`.
"""

Int: TypeAlias = Union[
    int,
    CustomType[pa.Int64Scalar],
    CustomType[pa.Int32Scalar],
    CustomType[pa.Int16Scalar],
    CustomType[pa.Int8Scalar],
    _Int64,
    _Int32,
    _Int16,
    _Int8,
]
"""
Int: Type alias for a signed integer of varying bit length.

Supported types include :class:`int`, :class:`pyarrow.Int8Scalar`, :class:`pyarrow.Int16Scalar`,
:class:`pyarrow.Int32Scalar`, :class:`pyarrow.Int64Scalar`, :class:`_Int8`, :class:`_Int16`,
:class:`_Int32`, and :class:`_Int64`.
"""

Float16: TypeAlias = Union[float, CustomType[pa.HalfFloatScalar], _Float16]
"""
Float16: Type alias for a 16-bit floating-point number.

Supported types include :class:`float`, :class:`pyarrow.HalfFloatScalar`, and :class:`_Float16`.
"""

Float32: TypeAlias = Union[float, CustomType[pa.FloatScalar], _Float32]
"""
Float32: Type alias for a 32-bit floating-point number.

Supported types include :class:`float`, :class:`pyarrow.FloatScalar`, and :class:`_Float32`.
"""

Float64: TypeAlias = Union[float, CustomType[pa.DoubleScalar], _Float64]
"""
Float64: Type alias for a 64-bit floating-point number.

Supported types include :class:`float`, :class:`pyarrow.DoubleScalar`, and :class:`_Float64`.
"""

Float: TypeAlias = Union[
    float,
    CustomType[pa.DoubleScalar],
    CustomType[pa.FloatScalar],
    CustomType[pa.HalfFloatScalar],
    _Float64,
    _Float32,
    _Float16,
]
"""
Float: Type alias for a floating-point number of varying precision.

Supported types include :class:`float`, :class:`pyarrow.HalfFloatScalar`,
:class:`pyarrow.FloatScalar`, :class:`pyarrow.DoubleScalar`, :class:`_Float16`,
:class:`_Float32`, and :class:`_Float64`.
"""
