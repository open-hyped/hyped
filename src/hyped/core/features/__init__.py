from typing import Annotated, Any, TypeAlias, Union

from .features import (
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
)
from .validators import Default, Len, TypeResolver, TypeValidator

__all__ = [
    "Union",
    "Annotated",
    "Feature",
    "String",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
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

String: TypeAlias = Annotated[Union[str, _String], Default(_String)]
Bool: TypeAlias = Annotated[Union[bool, _Bool], Default(_Bool)]

Int8: TypeAlias = Annotated[Union[int, _Int8], Default(_Int8)]
Int16: TypeAlias = Annotated[Union[int, _Int16], Default(_Int16)]
Int32: TypeAlias = Annotated[Union[int, _Int32], Default(_Int32)]
Int64: TypeAlias = Annotated[Union[int, _Int64], Default(_Int64)]
Int: TypeAlias = Annotated[Union[int, _Int8, _Int16, _Int32, _Int64], Default(_Int64)]

Float16: TypeAlias = Annotated[Union[float, _Float16], Default(_Float16)]
Float32: TypeAlias = Annotated[Union[float, _Float32], Default(_Float32)]
Float64: TypeAlias = Annotated[Union[float, _Float64], Default(_Float64)]
Float: TypeAlias = Annotated[Union[float, _Float16, _Float32, _Float64], Default(_Float64)]
