import inspect
from typing import Annotated, Callable, Iterable, ParamSpec, TypeVar

import pydantic

from hyped.common._features import type_hint_to_feature
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
    except ValueError as e:
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

        if not (
            hasattr(signature.return_annotation, "__args__")
            and (len(signature.return_annotation.__args__) > 0)
        ):
            raise ValueError(
                f"Cannot determine the item type of the iterable '{signature.return_annotation}'."
            )

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
