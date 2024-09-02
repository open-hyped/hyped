"""Utilities for handling feature references and constants in data flows.

This module provides functions to validate and convert constants to feature references, 
decorators for binary operations on feature references, and a method to collect 
features into a feature collection for further processing in data flow graphs.
"""
from functools import wraps
from typing import Any, Callable

import hyped.processors._ops as ops
from hyped.common._container import NestedContainer
from hyped.core.nodes.const import Const
from hyped.core.refs.ref import FeatureRef


def _check_args(*args: FeatureRef | Any) -> tuple[FeatureRef]:
    """Ensure at least one argument is a FeatureRef and convert constants to feature references.

    This function performs two main tasks:
    1. It checks that at least one of the arguments is a FeatureRef instance.
    2. It converts any non-FeatureRef arguments into FeatureRef instances by using the flow
       associated with the first FeatureRef in the sequence.

    Args:
        *args (FeatureRef | Any): A variable number of arguments, which can be either
            FeatureRef instances or constant values.

    Returns:
        tuple[FeatureRef]: A tuple where all constant values have been converted into
            FeatureRef instances, and the original FeatureRef instances are unchanged.

    Raises:
        RuntimeError: If all inputs are constants (i.e., there are no FeatureRef instances).
    """
    if not any(isinstance(a, FeatureRef) for a in args):
        raise RuntimeError(
            "All inputs are constants. At least one input must " "be a FeatureRef instance."
        )

    # get the flow from the argument sequence
    flow = next(iter(arg for arg in args if isinstance(arg, FeatureRef))).flow_

    # add all constants in the argument sequence to the flow
    return tuple(
        (arg if isinstance(arg, FeatureRef) else Const(value=arg).call(flow).value) for arg in args
    )


def _handle_constant_inputs_for_binary_op(
    binary_op: Callable[[FeatureRef, FeatureRef], FeatureRef]
) -> Callable[[FeatureRef | Any, FeatureRef | Any], FeatureRef]:
    """Decorator to handle constant inputs for binary operations on feature references.

    This decorator allows binary operations to be applied to a mix of feature references
    and constant values. If both inputs are constants, it raises an error. If one of the
    inputs is a constant, it is converted into a feature reference before the binary
    operation is applied.

    Args:
        binary_op (Callable[[FeatureRef, FeatureRef], FeatureRef]): The binary operation
            function to be decorated.

    Returns:
        Callable[[FeatureRef | Any, FeatureRef | Any], FeatureRef]: The wrapped binary operation
        function that can handle constant inputs.

    Raises:
        ValueError: If both inputs are constants.
    """

    @wraps(binary_op)
    def wrapped_binary_op(a: FeatureRef | Any, b: FeatureRef | Any) -> FeatureRef:
        # add constant arguments to the data flow
        a, b = _check_args(a, b)
        # apply binary operation on feature refs
        return binary_op(a, b)

    return wrapped_binary_op


def collect(
    collection: None | dict | list = None, flow: None | object = None, **kwargs
) -> FeatureRef:
    """Collects features into a feature collection.

    This function collects features into a feature collection, which can then be used as input
    to other nodes in the data flow graph. It accepts either a collection (dict or list) or keyword
    arguments representing feature values. If both collection and kwargs are provided,
    an error is raised.

    If any non-reference values are present in the collection, they are added as
    constants to the data flow graph.

    Args:
        collection (None | dict | list, optional): A collection (dict or list)
            containing features or feature values. Defaults to None.
        flow (None | object, optional): The data flow object. If not provided, the
            flow is inferred from the feature references in the collection. Defaults to None.
        **kwargs: Keyword arguments representing feature values.

    Returns:
        FeatureRef: A feature reference to the collected features.

    Raises:
        ValueError: If both collection and keyword arguments are provided.
        RuntimeError: If the flow cannot be inferred from the constant collection and no
            flow is provided explicitly.
    """
    if (collection is not None) and len(kwargs) > 0:
        raise ValueError(
            "Both `collection` and keyword arguments provided. " "Please provide only one."
        )

    # create a nested container from the inputs
    # this collection might contain constants of any type
    container = NestedContainer[FeatureRef | Any](
        data=collection if collection is not None else kwargs
    )

    if flow is None:
        # get the flow referenced in the collection in case it
        # contains any feature reference
        vals = container.flatten().values()
        vals = [v for v in vals if isinstance(v, FeatureRef)]

        if len(vals) == 0:
            raise RuntimeError(
                "Could not infer flow from constant collection, please "
                "specify the flow explicitly by setting the `flow` argument."
            )

        # get the flow from the first valid feature reference
        # in the nested collection
        flow = next(iter(vals)).flow_

    def _add_const(p: tuple[str, int], v: FeatureRef | Any) -> FeatureRef:
        return v if isinstance(v, FeatureRef) else Const(value=v).call(flow).value

    # add all constants in the collection to the flow
    container = container.map(_add_const, FeatureRef)

    return ops.CollectFeatures().call(collection=container).collected
