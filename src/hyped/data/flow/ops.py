"""Provides high-level feature operators for data processors.

The operator module defines high-level functions for performing common operations.
These functions delegate the actual processing to specific processor classes and
return references to the resulting features represented by `FeatureRef` instances.

Feature operators are designed to simplify the process of adding processors to a data
flow by providing high-level functions for common feature operations. Each operator
encapsulates the logic for performing specific tasks, such as collecting features from
a collection (e.g., dictionary or list). These functions leverage underlying processor
classes, such as `CollectFeatures`, to execute the desired operations.

Functions:
    - :class:`collect`: Collect features from a given collection.

Usage Example:
    Collect features from a dictionary using the :class:`collect` operator:

    .. code-block:: python

        # Import the collect operator from the module
        from hyped.data.processors.operator import collect

        # Define the features of the source node
        src_features = datasets.Features({"text": datasets.Value("string")})

        # Initialize a DataFlow instance with the source features
        flow = DataFlow(features=src_features)
        
        # Collect features from the dictionary using the collect operator
        collected_features = collect(
            collection={
                "out": [
                    flow.src_features.text,
                    flow.src_features.text
                ]
            }
        )

        collected_features.out  # work with the collected features

"""

from .aggregators.ops.mean import MeanAggregator
from .aggregators.ops.sum import SumAggregator
from .aggregators.ref import DataAggregationRef
from .processors.ops import binary
from .processors.ops.collect import CollectFeatures
from .refs.ref import FeatureRef


def collect(collection: None | dict | list = None, **kwargs) -> FeatureRef:
    """Collect features from a given collection.

    This function provides a high-level operator for collecting features from a given collection,
    such as a dictionary or a list. It delegates the collection process to the :class:`CollectFeatures`
    class and returns a :class:`FeatureRef` instance representing the collected features.

    Args:
        collection (None | dict | list, optional): The collection from which to collect features.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the collection process.

    Returns:
        FeatureRef: A FeatureRef instance representing the collected features.
    """
    return CollectFeatures().call(collection=collection, **kwargs).collected


def sum_(a: FeatureRef) -> DataAggregationRef:
    """Calculate the sum of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        DataAggregationRef: A reference to the result of the sum operation.
    """
    return SumAggregator().call(x=a)


def mean(a: FeatureRef) -> DataAggregationRef:
    """Calculate the mean of feature values.

    Args:
        a (FeatureRef): The feature to aggregate.

    Returns:
        DataAggregationRef: A reference to the result of the mean operation.
    """
    return MeanAggregator().call(x=a)


def add(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Add two features.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the addition.
    """
    return binary.Add().call(a=a, b=b).result


def sub(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Subtract one feature from another.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the subtraction.
    """
    return binary.Sub().call(a=a, b=b).result


def mul(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Multiply two features.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the multiplication.
    """
    return binary.Mul().call(a=a, b=b).result


def truediv(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Divide one feature by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the division.
    """
    return binary.TrueDiv().call(a=a, b=b).result


def floordiv(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform integer division of one feature by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the integer division.
    """
    return binary.FloorDiv().call(a=a, b=b).result


def pow(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Raise one feature to the power of another.

    Args:
        a (FeatureRef): The base feature.
        b (FeatureRef): The exponent feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the exponentiation.
    """
    return binary.Pow().call(a=a, b=b).result


def mod(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Calculate the modulo of one features by another.

    Args:
        a (FeatureRef): The dividend feature.
        b (FeatureRef): The divisor feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the modulo operation.
    """
    return binary.Mod().call(a=a, b=b).result


def eq(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if two features are equal.

    Args:
        a (FeatureRef): The first feature
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the equality comparison.
    """
    return binary.Equals().call(a=a, b=b).result


def ne(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if two feature references are not equal.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the inequality comparison.
    """
    return binary.NotEquals().call(a=a, b=b).result


def lt(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is less than the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the less-than comparison.
    """
    return binary.LessThan().call(a=a, b=b).result


def le(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is less than or equal to the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the less-than-or-equal-to comparison.
    """
    return binary.LessThanOrEqual().call(a=a, b=b).result


def gt(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is greater than the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the greater-than comparison.
    """
    return binary.GreaterThan().call(a=a, b=b).result


def ge(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Check if the first feature is greater than or equal to the second.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the greater-than-or-equal-to comparison.
    """
    return binary.GreaterThanOrEqual().call(a=a, b=b).result


def and_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical AND operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the and operation.
    """
    return binary.LogicalAnd().call(a=a, b=b).result


def or_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical OR operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the or operation.
    """
    return binary.LogicalOr().call(a=a, b=b).result


def xor_(a: FeatureRef, b: FeatureRef) -> FeatureRef:
    """Perform a logical XOR operation on two feature.

    Args:
        a (FeatureRef): The first feature.
        b (FeatureRef): The second feature.

    Returns:
        FeatureRef: A FeatureRef instance representing the result of the xor operation.
    """
    return binary.LogicalXOr().call(a=a, b=b).result
