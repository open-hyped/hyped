"""Provides binary operations on feature references.

This module provides high-level binary operations on feature references, enabling
arithmetic, comparison, and logical operations within a data flow. The operations 
handle both feature references and constant values, ensuring seamless integration 
of different data types.

Each binary operation, such as addition (:func:`add`), subtraction (:func:`sub`), 
and logical AND (:func:`and_`), is implemented as a function that can be directly 
applied to features in the data flow. These functions are decorated to support 
constant values, converting them into feature references when necessary.
"""
from functools import wraps
from typing import Any, Callable

import hyped.processors._ops as ops
from hyped.core.refs.ref import FeatureRef
from hyped.ops.utils import _handle_constant_inputs_for_binary_op


@_handle_constant_inputs_for_binary_op
def add(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Add two features.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the addition.
    """
    return ops.Add().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def sub(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Subtract one feature from another.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the subtraction.
    """
    return ops.Sub().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def mul(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Multiply two features.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the multiplication.
    """
    return ops.Mul().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def truediv(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Divide one feature by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the division.
    """
    return ops.TrueDiv().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def floordiv(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform integer division of one feature by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the integer division.
    """
    return ops.FloorDiv().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def pow_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Raise one feature to the power of another.

    Args:
        a (FeatureRef): The base feature.
        b (FeatureRef): The exponent feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the exponentiation.
    """
    return ops.Pow().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def mod(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Calculate the modulo of one features by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the modulo operation.
    """
    return ops.Mod().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def eq(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if two features are equal.

    Args:
        a (FeatureRef): The first feature
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the equality comparison.
    """
    return ops.Equals().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def ne(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if two feature references are not equal.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the inequality comparison.
    """
    return ops.NotEquals().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def lt(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is less than the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the less-than comparison.
    """
    return ops.LessThan().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def le(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is less than or equal to the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the
            less-than-or-equal-to comparison.
    """
    return ops.LessThanOrEqual().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def gt(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is greater than the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the greater-than comparison.
    """
    return ops.GreaterThan().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def ge(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is greater than or equal to the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the
            greater-than-or-equal-to comparison.
    """
    return ops.GreaterThanOrEqual().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def and_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical AND operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the and operation.
    """
    return ops.LogicalAnd().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def or_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical OR operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the or operation.
    """
    return ops.LogicalOr().call(a=a, b=b).result


@_handle_constant_inputs_for_binary_op
def xor_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical XOR operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the xor operation.
    """
    return ops.LogicalXOr().call(a=a, b=b).result
