"""Provides binary operations on feature references.

This module provides high-level binary operations on features, enabling
arithmetic, comparison, and logical operations within a data flow.
"""

from typing import Callable, TypeVar

import hyped.nodes._ops as ops
from hyped.core.features.features import (
    Feature,
    Float16Feature,
    Float32Feature,
    Float64Feature,
    Int8Feature,
    Int16Feature,
    Int32Feature,
    Int64Feature,
    UInt8Feature,
    UInt16Feature,
    UInt32Feature,
    UInt64Feature,
)

ScalarType = TypeVar("ScalarType")

# TODO: implement casting and broadcasting here

Fn = TypeVar("Fn", bound=Callable)


def register_all(name: str, types: type[Feature]) -> Callable[[Fn], Fn]:
    def wrapper(fn):
        for feature_type in types:
            feature_type.register_method(name)(fn)
        return fn

    return wrapper


@register_all(
    "__add__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def add(a: ScalarType, b: ScalarType) -> ScalarType:
    return ops.Add().call(a, b)


@register_all(
    "__sub__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def sub(a: ScalarType, b: ScalarType) -> ScalarType:
    return ops.Subtract().call(a, b)


@register_all(
    "__mul__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def mul(a: ScalarType, b: ScalarType) -> ScalarType:
    return ops.Multiply().call(a, b)


@register_all(
    "__truediv__",
    [
        Int8Feature,
        Int16Feature,
        Int32Feature,
        Int64Feature,
        UInt8Feature,
        UInt16Feature,
        UInt32Feature,
        UInt64Feature,
        Float16Feature,
        Float32Feature,
        Float64Feature,
    ],
)
def div(a: ScalarType, b: ScalarType) -> ScalarType:
    return ops.Divide().call(a, b)
