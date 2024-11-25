"""Provides base classe for data processors in a data flow graph.

This module defines the base class for data processors, which represent
nodes in a data flow graph. It includes generic classes for defining
data processors with configurable input and output types.
"""
from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import (
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import pyarrow as pa
from typing_extensions import Self

from ..typing import Feature
from .base import BaseNode, BaseNodeConfig, NodeProtocol, ProcessMode, RunContext

Params = ParamSpec("Params")
Return = TypeVar("Return", covariant=True)


@runtime_checkable
class _ProcessFunctionProtocol(Protocol, Generic[Params, Return]):
    def process(self, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        ...  # pragma: not covered


class BaseDataProcessorConfig(BaseNodeConfig):
    """Base configuration class for data processors.

    This class serves as the base configuration class for data processors.
    It inherits from :code:`BaseNodeConfig`, a Pydantic model, providing
    basic configuration functionality for data processing tasks.
    """


C = TypeVar("C", bound=BaseDataProcessorConfig)


class BaseDataProcessor(BaseNode[C], ABC):
    """Base class for data processors in a data flow graph.

    This class serves as the base for all data processors, representing nodes in a data flow graph.
    Subclasses of :class:`BaseDataProcessor` implement specific process functions that map input
    features to output features. Custom data processors must either override the
    :func:`batch_process` method or the :func:`process` method to define their processing logic.
    """

    def __new__(
        cls: _ProcessFunctionProtocol[Concatenate[Self, RunContext, Params], Return],
        *args: Any,
        **kwargs: Any,
    ) -> NodeProtocol[Params, Return]:
        """Create a new instance of the data processor node.

        This method is responsible for creating a new instance of the data processor in
        the data flow graph, ensuring it conforms to the node protocol.

        Args:
            *args (Any): Positional arguments for initializing the processor node.
            **kwargs (Any): Keyword arguments for initializing the processor node.

        Returns:
            NodeProtocol[Params, Return]: An instance of the processor node in the data flow graph.
        """
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def _check_signature(cls) -> bool:
        """Validate that the class conforms to the :class:`_ProcessFunctionProtocol`.

        This method ensures that the subclass implements a :code:`process` method
        with the correct signature as required by the :class:`_ProcessFunctionProtocol`.

        Returns:
            bool: `True` if the class conforms to the protocol; `False` otherwise.
        """
        # check the protocol, in practice this can never fire because
        # the process function is abstract
        if not issubclass(cls, _ProcessFunctionProtocol):
            return False  # pragma: not covered
        # get the process function parameters
        params = inspect.signature(cls.process).parameters
        # check if the context argument is the first argument
        # after self and has the correct annotation
        return (
            ("ctx" in params)
            and (list(params.keys()).index("ctx") == 1)
            and (params["ctx"].annotation is RunContext)
        )

    @classmethod
    def __init_subclass__(cls) -> None:
        """Initialize a new subclass and enforce protocol adherence.

        This method is called automatically whenever a class inherits from
        :class:`BaseDataProcessor`. The workflow is as follows:

        1. Sets the default processing mode to the :code:`process` method.
        2. Validates that the subclass conforms to the :class:`_ProcessFunctionProtocol`.
        3. Raises a :class:`TypeError` if the subclass does not meet the protocol requirements.

        Raises:
            TypeError: If the subclass does not implement a :code:`process` method
            that matches the :class:`_ProcessFunctionProtocol` signature.
        """
        # apply default process mode, only sets the process mode if the
        # function doesn't have a process mode applied to it yet
        ProcessMode(batched=False, backend="python").validate().set_default(cls.process)

        if not cls._check_signature():
            raise TypeError(
                f"The class '{cls.__name__}' must implement a 'process' method that matches "
                "the signature defined in the '_ProcessFunctionProtocol'. Ensure the method's "
                "arguments and return type conform to the expected protocol."
            )

    def __init__(self, config: None | C = None, **kwargs) -> None:
        """Initialize the data processor.

        Initializes the data processor with the given configuration. If no configuration is
        provided, a new configuration is created using the provided keyword arguments.

        Args:
            config (C, optional): The configuration object for the data processor. If not provided,
                a configuration is created based on the given keyword arguments.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(BaseDataProcessor, self).__init__(config, **kwargs)
        # check whether the process function is a coroutine
        self._is_process_async = inspect.iscoroutinefunction(self.process)

    @property
    def signature(self) -> inspect.Signature:
        """Get the signature of the :func:`process` method.

        Returns the signature of the :func:`process` method with the :code:`ctx` parameter
        removed, keeping only the feature inputs modeled in the data flow graph.

        Returns:
            inspect.Signature: The signature of the :func:`process` method excluding
            the :code:`ctx` parameter.
        """
        signature = inspect.signature(self.process)
        # remove the ctx argument of the process function
        params = signature.parameters.values()
        params = [param for param in params if param.name != "ctx"]
        # return the signature containing the remaining parameters
        return signature.replace(parameters=params)

    @overload
    async def process(self, ctx: RunContext, *args: Feature, **kwargs: Feature) -> Feature:
        ...

    @abstractmethod
    def process(self, ctx: RunContext, *args: Feature, **kwargs: Feature) -> Feature:
        """Process a single data sample.

        This method should be implemented by subclasses to define the processing of a single
        data sample. It may either be synchronous or asynchronous, depending on the subclass.

        It also defines the interface of the node and must be implemented by subclasses. Even if
        the :func:`batch_process` method implements the primary processing logic, :func:`process`
        is still required to define the input and output features of the node.

        Args:
            ctx (RunContext): The context for the current process call.
            *args (Feature): Positional feature arguments.
            **kwargs (Feature): Keyword feature arguments.

        Returns:
            Feature: The resulting processed feature.
        """
        ...

    async def run(self, ctx: RunContext, arrays: dict[str, pa.Array]) -> pa.Array:
        """Execute the main processing logic for the data processor.

        This method serves as the primary entry point for processing data within
        a data flow graph. It orchestrates the execution of the :code:`process`
        method according to the configured :class:`ProcessMode`, handling input
        preparation, processing, and output finalization. In detail the workflow
        is:

        1. Determine the processing mode (:class:`ProcessMode`) based on the
           :code:`process` method's configuration.
        2. Prepare the input data using the :class:`ProcessMode.prepare` method.
        3. Apply the :code:`process` method to the prepared inputs. If the method
           is asynchronous, the outputs are awaited using :code:`asyncio.gather`.
        4. Finalize the outputs using the :code:`ProcessMode.finalize` method,
           which ensures that the results are correctly formatted as an
           :code:`PyArrow` array.

        Args:
            ctx (RunContext): The execution context containing.
            arrays (dict[str, pa.Array]): A dictionary mapping input names to
                :code:`PyArrow` arrays, representing the input data to be processed.

        Returns:
            pa.Array: The processed output as a :code:`PyArrow` array.
        """
        # get the process mode
        mode = ProcessMode.from_decorated_fn(self.process)

        # prepare inputs and apply process function to all inputs
        inputs = mode.prepare(ctx, **arrays)
        outputs = (self.process(c, **kw) for c, kw in inputs)

        # gather all outputs in case the process function is a coroutine
        if self._is_process_async:
            outputs = await asyncio.gather(*outputs)

        # finalize outputs
        return mode.finalize(ctx, outputs)
