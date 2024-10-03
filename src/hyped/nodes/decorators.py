"""This module provides decorators for transforming user-defined functions into nodes.

It validates function signatures, generates appropriate input/output reference types, and
dynamically creates specialized subclasses for processing data in workflows. The decorators
handle synchronous and asynchronous functions, support type annotations for parameters and
return values, and ensure compatibility with specific pipeline stages by enforcing required
function signature constraints.
"""

import inspect
from types import UnionType
from typing import Annotated, Callable, Iterable, ParamSpec, TypeVar, Union, get_args, get_origin

import pydantic

from hyped.common._features import type_hint_to_feature
from hyped.common.utils import dict_of_lists_to_list_of_dicts
from hyped.core.nodes.aggregator import BaseDataAggregator, BaseDataAggregatorConfig
from hyped.core.nodes.augmenter import BaseDataAugmenter, BaseDataAugmenterConfig
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.core.refs.outputs import OutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef


def _create_in_out_types(
    prefix: str, params: dict[str, inspect.Parameter], return_annotation: inspect.Parameter
) -> tuple[type[InputRefs], type[OutputRefs]]:
    """Generates input and output reference types for a given function.

    This helper function validates the function parameters and return type,
    ensuring that all parameters have type annotations. It creates types for
    input references based on the parameter annotations and for output
    references based on the return type annotation.

    Args:
        prefix (str): A prefix for naming the generated types.
        params (dict[str, inspect.Parameter]): A dictionary of function parameters.
        return_annotation (inspect.Parameter): The return type annotation of the function.

    Returns:
        tuple[type[InputRefs], type[OutputRefs]]: A tuple containing the generated
        input references type and output references type.

    Raises:
        ValueError: If the function includes *args or **kwargs.
        ValueError: If parameters or return types are missing annotations.
    """

    # Check for *args or **kwargs
    if any(
        param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for param in params.values()
    ):
        raise ValueError("The function should not include *args or **kwargs.")

    input_dict = {}
    for name, param in params.items():
        # make sure the parameter has a type annotation
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(f"The parameter '{name}' in function does not have a type hint.")
        # create type validator from type hint
        feature = type_hint_to_feature(param.annotation)
        validator = CheckFeatureEquals(feature)
        # add type validator to input annotations
        input_dict[name] = Annotated[FeatureRef, validator]

    # check return annotation
    if return_annotation is inspect.Signature.empty:
        raise ValueError("The function does not have a return type annotation.")
    # convert the output type annotation to a dataset feature
    output_feature = type_hint_to_feature(return_annotation)
    output_annotation = Annotated[FeatureRef, OutputFeature(output_feature)]

    # create the configuration, input refs and output refs type
    inputs_t = type(f"{prefix}_input_refs", (InputRefs,), {"__annotations__": input_dict})
    output_t = pydantic.create_model(
        f"{prefix}_output_refs",
        __base__=OutputRefs,
        output=(output_annotation, None),
    )

    return inputs_t, output_t


P = ParamSpec("P")
T = TypeVar("T")


def as_processor(func: Callable[P, T]) -> Callable[P, FeatureRef]:
    """Wraps a user-defined function to create a data processor.

    This decorator transforms a given function into a data processor by generating
    appropriate configuration and input/output reference types. The wrapped function
    must have type annotations for its parameters and return value. It cannot use
    variable positional (`*args`) or keyword arguments (`**kwargs`).

    The decorator performs the following steps:
    1. Validates that all parameters have type annotations.
    2. Constructs input reference types based on the function's parameters.
    3. Constructs output reference types based on the function's return type.
    4. Creates a new `BaseDataProcessor` subclass that implements the `process`
        method to call the wrapped function.

    Args:
        func (Callable[P, T]): The function to be wrapped. This function should
            define its input parameters and return type with appropriate type hints.

    Returns:
        Callable[P, FeatureRef]: A callable that, when invoked, creates an instance
        of the generated processor class and executes the wrapped function with
        the provided inputs, returning the output in the required format.

    Raises:
        ValueError: If the function includes `*args` or `**kwargs`.
        ValueError: If any of the function parameters are missing type annotations.
        ValueError: If the function return type is not annotated.
    """

    signature = inspect.signature(func)
    # create config type
    config_t = type(f"{func.__name__}_config", (BaseDataProcessorConfig,), {})

    try:
        # create the config, input refs input output refs types
        inputs_t, output_t = _create_in_out_types(
            func.__name__, signature.parameters, signature.return_annotation
        )
    except ValueError as e:  # pragma: not covered
        raise ValueError(f"Error parsing function '{func.__qualname__}'.") from e

    if inspect.iscoroutinefunction(func):

        class AsyncFunctionProcessor(BaseDataProcessor[config_t, inputs_t, output_t]):
            __name__ = f"{func.__name__}_processor"

            async def process(self, inputs, index, rank, io):
                return {"output": await func(**inputs)}

        return lambda *args, **kwargs: AsyncFunctionProcessor().call(*args, **kwargs).output

    else:

        class FunctionProcessor(BaseDataProcessor[config_t, inputs_t, output_t]):
            __name__ = f"{func.__name__}_processor"

            def process(self, inputs, index, rank, io):
                return {"output": func(**inputs)}

        return lambda *args, **kwargs: FunctionProcessor().call(*args, **kwargs).output


P = ParamSpec("P")
T = TypeVar("T")


def as_augmenter(func: Callable[P, T]) -> Callable[P, FeatureRef]:
    """Wraps a user-defined generator function to create a data augmenter.

    This decorator transforms a given generator function into a data augmenter
    by generating appropriate configuration and input/output reference types.
    The wrapped function must have type annotations for its parameters and
    return an iterable of items. It cannot use variable positional (`*args`)
    or keyword arguments (`**kwargs`).

    The decorator performs the following steps:
    1. Validates that all parameters have type annotations.
    2. Ensures that the function is a generator function.
    3. Validates that the return type is an iterable and extracts its item type.
    4. Constructs input reference types based on the function's parameters.
    5. Constructs output reference types based on the item type of the return value.
    6. Creates a new `BaseDataAugmenter` subclass that implements the `process`
       method to yield processed outputs from the wrapped generator function.

    Args:
        func (Callable[P, T]): The generator function to be wrapped. This function should
            define its input parameters and return type with appropriate type hints.

    Returns:
        Callable[P, FeatureRef]: A callable that, when invoked, creates an instance
        of the generated augmenter class and processes the inputs, yielding the output
        in the required format.

    Raises:
        ValueError: If the function is not a generator function.
        ValueError: If the function return type is not an iterable type.
        ValueError: If the item type of the return iterable cannot be determined.
        ValueError: If any of the function parameters are missing type annotations.
    """

    signature = inspect.signature(func)
    # create config type
    config_t = type(f"{func.__name__}_config", (BaseDataAugmenterConfig,), {})

    try:
        if (not inspect.isgeneratorfunction(func)) and (not inspect.isasyncgenfunction(func)):
            raise ValueError(f"The function '{func.__qualname__}' must be a generator function.")

        if not isinstance(signature.return_annotation, Iterable):
            raise ValueError(
                f"The return type of the function '{func.__qualname__}' must be an "
                f"iterable type, but got '{signature.return_annotation}'."
            )

        assert hasattr(signature.return_annotation, "__args__")
        assert len(signature.return_annotation.__args__) > 0

        # create the config, input refs input output refs types
        inputs_t, output_t = _create_in_out_types(
            func.__name__, signature.parameters, signature.return_annotation.__args__[0]
        )
    except ValueError as e:
        raise ValueError(f"Error parsing function '{func.__qualname__}'.") from e

    if inspect.isasyncgenfunction(func):

        class AsyncFunctionAugmenter(BaseDataAugmenter[config_t, inputs_t, output_t]):
            __name__ = f"{func.__name__}_augmenter"

            async def process(self, inputs, index, rank, io):
                async for val in func(**inputs):
                    yield {"output": val}

        return lambda *args, **kwargs: AsyncFunctionAugmenter().call(*args, **kwargs).output

    else:

        class FunctionAugmenter(BaseDataAugmenter[config_t, inputs_t, output_t]):
            __name__ = f"{func.__name__}_augmenter"

            def process(self, inputs, index, rank, io):
                yield from ({"output": val} for val in func(**inputs))

        return lambda *args, **kwargs: FunctionAugmenter().call(*args, **kwargs).output


P = ParamSpec("P")
T = TypeVar("T")


def as_aggregator(func: Callable[P, T]) -> Callable[P, FeatureRef]:
    """Wraps a user-defined function to create a data aggregator.

    This decorator transforms a given function into a data aggregator by generating
    appropriate configuration and input/output reference types. The wrapped function
    must follow specific signature constraints: it must accept :code:`value` and
    :code:`ctx` as arguments, and its return type must be a tuple containing exactly
    two items, representing the aggregated value and aggregation context.

    The decorator performs the following steps:
    1. Ensures the function includes :code:`value` and :code:`ctx` parameters.
    2. Validates that the return type is a tuple with exactly two elements: the value
       and the context.
    3. Ensures that the :code:`value` parameter is an optional type that matches the first
       element of the return tuple.
    4. Ensures that the :code:`ctx` parameter matches the second element of the return tuple.
    5. Constructs input reference types based on the function's remaining parameters.
    6. Constructs output reference types based on the return type.

    Args:
        func (Callable[P, T]): The function to be wrapped. This function should define
            its input parameters (including :code:`value` and :code:`ctx`) and return type
            with appropriate type hints.

    Returns:
        Callable[P, FeatureRef]: A callable that, when invoked, creates an instance of
        the generated aggregator class and executes the wrapped function with the
        provided inputs, returning the aggregated value and updated context.

    Raises:
        ValueError: If the function does not include :code:`value` or :code:`ctx` as parameters.
        ValueError: If the function's return type is not a tuple with exactly two elements.
        ValueError: If the :code:`value` parameter type does not match the first element of the
            return tuple.
        ValueError: If the :code:`ctx` parameter type does not match the second element of the
            return tuple.

    Example:
        Here is an example of using :func:`as_aggregator` to compute the average value of an
        integer.

        .. code-block:: python

            from typing import Optional

            @as_aggregator
            def f(value: Optional[float], ctx: Optional[int], x: int) -> tuple[float, int]:
                \"\"\"
                Aggregates the average of a sequence of integers.

                Args:
                    value (Optional[float]): The current aggregated value (or None if starting).
                    ctx (Optional[int]): The current count of aggregated values.
                    x (int): The next integer to include in the aggregation.

                Returns:
                    tuple[float, int]: The updated average and the new count.
                \"\"\"

                if value is None:
                    # initialize the value and aggregation context
                    value = 0
                    ctx = 0

                # compute the running average and increment the count
                return (ctx * value + x) / (ctx + 1), ctx + 1

            # Now, calling f would use the aggregator logic.
    """

    signature = inspect.signature(func)
    # create config type
    config_t = type(f"{func.__name__}_config", (BaseDataAggregatorConfig,), {})

    try:
        # must contain value and state
        if ("value" not in signature.parameters) or ("ctx" not in signature.parameters):
            raise ValueError("The function must include 'value' and 'ctx' parameters.")

        # Get the origin and args of the return annotation
        origin = get_origin(signature.return_annotation)
        args = get_args(signature.return_annotation)
        # Check if the return annotation is a tuple and has exactly two items
        if origin is not tuple or len(args) != 2:
            raise ValueError(
                "The return type of the function must be a tuple containing exactly two "
                "items, namely the value and the context."
            )

        # get the annotation of the value
        value_return_annotation, ctx_return_annotation = args

        # make sure the input value annotation matches the value return annotation
        value_origin = get_origin(signature.parameters["value"].annotation)
        value_args = get_args(signature.parameters["value"].annotation)

        if not (
            (value_origin in (Union, UnionType))
            and (value_return_annotation in value_args)
            and (type(None) in value_args)
        ):
            raise ValueError(
                "The 'value' parameter must be of the same type as the first element of "
                "the return type and must be an optional type (i.e., Union[Type, None])."
            )

        if signature.parameters["ctx"].annotation is not None:
            # make sure the input value annotation matches the value return annotation
            ctx_origin = get_origin(signature.parameters["ctx"].annotation)
            ctx_args = get_args(signature.parameters["ctx"].annotation)

            if not (
                (ctx_origin in (Union, UnionType))
                and (ctx_return_annotation in ctx_args)
                and (type(None) in ctx_args)
            ):
                raise ValueError(
                    "The 'ctx' parameter must be of the same type as the second element of "
                    "the return type and must be an optional type (i.e., Union[Type, None])."
                )

        elif ctx_return_annotation is not None:
            # make sure the input state annotation matches the state return annotation
            raise ValueError(
                "The 'ctx' parameter type hint must match the second element of the return type."
            )

        # get remaining params
        params = signature.parameters.copy()
        params.pop("value")
        params.pop("ctx")

        # create the config, input refs input output refs types
        inputs_t, output_t = _create_in_out_types(
            func.__name__, params, signature.return_annotation.__args__[0]
        )

    except ValueError as e:
        raise ValueError(f"Error parsing function '{func.__qualname__}'.") from e

    if inspect.iscoroutinefunction(func):

        class AsyncFunctionAggregator(BaseDataAggregator[config_t, inputs_t, output_t]):
            __name__ = f"{func.__name__}_aggregator"

            def initialize(self, io):
                return {"output": None}, None

            async def extract(self, inputs, index, rank, io):
                return inputs

            async def update(self, val, state, ctx, io):
                val = val["output"]
                for item in dict_of_lists_to_list_of_dicts(state):
                    val, ctx = await func(value=val, ctx=ctx, **item)
                return {"output": val}, ctx

        return lambda *args, **kwargs: AsyncFunctionAggregator().call(*args, **kwargs).output

    else:

        class FunctionAggregator(BaseDataAggregator[config_t, inputs_t, output_t]):
            __name__ = f"{func.__name__}_aggregator"

            def initialize(self, io):
                return {"output": None}, None

            async def extract(self, inputs, index, rank, io):
                return inputs

            async def update(self, val, state, ctx, io):
                val = val["output"]
                for item in dict_of_lists_to_list_of_dicts(state):
                    val, ctx = func(value=val, ctx=ctx, **item)
                return {"output": val}, ctx

        return lambda *args, **kwargs: FunctionAggregator().call(*args, **kwargs).output  #
