"""This module defines a type cast operator."""

from typing import TYPE_CHECKING, Any
from typing import cast as typing_cast

from ..features.features import Feature, build_feature_from_annotation, build_feature_from_reference
from ..features.reference import ConcreteReference


def _cast(typ: Any, val: Any) -> Any:
    """Cast a feature to a specified data type.

    This function performs a type cast for a value if it is an instance of a
    :class:`Feature`. It infers the target data type from the provided type
    annotation, adds a cast node to the data flow graph, and retrieves the
    resulting feature.

    Args:
        typ (Any): The target type to cast the value to. Typically, this is a type
            annotation.
        val (Any): The value to be cast. If the value is not a :class:`Feature`, it
            is returned unchanged.

    Returns:
        Any: The cast value. If the input :code:`val` is a :class:`Feature`, the function
        returns the feature resulting from the cast operation. Otherwise, it returns :code:`val`
        unchanged.
    """
    # make sure the value is a feature
    if not isinstance(val, Feature):
        return val

    assert isinstance(val.ref, ConcreteReference)
    # infer the target dtype from the given type annotation and
    # add the cast node to the graph
    dtype = build_feature_from_annotation(typ).dtype
    ref = val.ref._builder.cast_node(val.ref, dtype)
    # return the output feature of the cast operation
    return build_feature_from_reference(ref)


cast = typing_cast if TYPE_CHECKING else _cast
"""Cast a feature to a specified data type.

This function performs a type cast for a value if it is an instance of a
:class:`Feature`. It infers the target data type from the provided type
annotation, adds a cast node to the data flow graph, and retrieves the
resulting feature.

Args:
    typ (Any): The target type to cast the value to. Typically, this is a type
        annotation.
    val (Any): The value to be cast. If the value is not a :class:`Feature`, it
        is returned unchanged.

Returns:
    Any: The cast value. If the input :code:`val` is a :class:`Feature`, the function
    returns the feature resulting from the cast operation. Otherwise, it returns :code:`val`
    unchanged.
"""
