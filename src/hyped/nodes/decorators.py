import inspect
from typing import Annotated, Callable, ParamSpec, TypeVar

import pydantic

from hyped.common._features import type_hint_to_feature
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.core.refs.outputs import OutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef

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
    # Check for *args or **kwargs
    if any(
        param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for param in signature.parameters.values()
    ):
        raise ValueError(f"The function {func.__qualname__} should not include *args or **kwargs.")

    input_dict = {}
    for name, param in signature.parameters.items():
        # make sure the parameter has a type annotation
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(
                f"The parameter '{name}' in function '{func.__name__}' does not have a type hint."
            )
        # create type validator from type hint
        feature = type_hint_to_feature(param.annotation)
        validator = CheckFeatureEquals(feature)
        # add type validator to input annotations
        input_dict[name] = Annotated[FeatureRef, validator]

    # check return annotation
    if signature.return_annotation is inspect.Signature.empty:
        raise ValueError(f"The function '{func.__name__}' does not have a return type annotation.")
    # convert the output type annotation to a dataset feature
    output_feature = type_hint_to_feature(signature.return_annotation)
    output_annotation = Annotated[FeatureRef, OutputFeature(output_feature)]

    # create the configuration, input refs and output refs type
    config_t = type(f"{func.__name__}_function_processor_config", (BaseDataProcessorConfig,), {})
    inputs_t = type(
        f"{func.__name__}_processor_input_refs",
        (InputRefs,),
        {"__annotations__": input_dict},
    )
    output_t = pydantic.create_model(
        f"{func.__name__}_processor_output_refs",
        __base__=OutputRefs,
        output=(output_annotation, None),
    )

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
