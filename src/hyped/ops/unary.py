"""Provides unary operations on feature references.

This module provides high-level unary operations for feature references within a
data flow. Unary operations involve a single operand and include common operations 
like calculating the sum (:func:`sum_`), mean (:func:`mean`), negation (:func:`neg`), 
absolute value (:func:`abs_`), and bitwise inversion (:func:`invert`).

Each operation is implemented as a function that can be applied directly to a 
feature reference. The module includes checks to determine whether a feature 
is a sequence, ensuring that appropriate processors are used for sequence 
features versus scalar features.
"""
from datasets import Value

import hyped.aggregators._ops as agg_ops
import hyped.processors._ops as proc_ops
from hyped.common.feature_checks import (
    check_feature_equals,
    check_feature_is_sequence,
)
from hyped.core.refs.ref import FeatureRef


def sum_(a: FeatureRef) -> FeatureRef:
    """Calculate the sum of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        FeatureRef: A reference to the result of the sum operation.
    """
    if check_feature_is_sequence(a.feature_):
        return proc_ops.SequenceSum().call(a=a).result
    else:
        return agg_ops.SumAggregator().call(x=a).value


def mean(a: FeatureRef) -> FeatureRef:
    """Calculate the mean of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        FeatureRef: A reference to the result of the mean operation.
    """
    if check_feature_is_sequence(a.feature_):
        return proc_ops.SequenceMean().call(a=a).result
    else:
        return agg_ops.MeanAggregator().call(x=a).value


def neg(a: FeatureRef) -> FeatureRef:
    """Perform a negation operation on a feature.

    Args:
        a (FeatureRef): The feature to negate.

    Returns:
        FeatureRef: A FeatureRef instance representing the negated value of the input feature.
    """
    return proc_ops.Neg().call(a=a).result


def abs_(a: FeatureRef) -> FeatureRef:
    """Compute the absolute value of a feature.

    Args:
        a (FeatureRef): The feature to compute the absolute value.

    Returns:
        FeatureRef: A FeatureRef instance representing the absolute value of the input feature.
    """
    return proc_ops.Abs().call(a=a).result


def invert(a: FeatureRef) -> FeatureRef:
    """Perform a bitwise inversion operation on a feature.

    Args:
        a (FeatureRef): The feature to invert bitwise.

    Returns:
        FeatureRef: A FeatureRef instance representing the bitwise inverted
            value of the input feature.
    """
    if check_feature_equals(
        a.feature_, Value("bool")
    ) or check_feature_is_sequence(a.feature_, Value("bool")):
        return proc_ops.BooleanInvert().call(a=a).result
    else:
        return proc_ops.Invert().call(a=a).result
