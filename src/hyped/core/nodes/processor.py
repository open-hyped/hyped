"""Provides base classe for data processors in a data flow graph.

This module defines the base class for data processors, which represent
nodes in a data flow graph. It includes generic classes for defining
data processors with configurable input and output types.
"""
from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Concatenate, Generic, ParamSpec, Protocol, TypeVar, overload

import pyarrow as pa
from typing_extensions import Self

from ..typing import Feature
from .base import BaseNode, BaseNodeConfig, NodeProtocol, RunContext

Params = ParamSpec("Params")
Return = TypeVar("Return", covariant=True)


class _ProcessFunctionProtocol(Protocol, Generic[Params, Return]):
    def process(self, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        ...


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

    async def batch_process(self, ctx: RunContext, **kwargs: pa.Array) -> pa.Array:
        """Process a batch of data samples.

        Applies the :func:`process` function to each sample in the batch and gathers the
        outputs, supporting asynchronous processing if :func:`process` is a coroutine.

        Args:
            ctx (RunContext): The context for the current batch processing call, including the
                input and output types, node ID, and index.
            **kwargs (pa.Array): Keyword arguments representing columns of data samples in the
                batch.

        Returns:
            pa.Array: The processed output batch as an Arrow array.

        Raises:
            RuntimeError: If the flow cannot be inferred from arguments.
        """
        # apply process function to each sample in the input batch
        batch = pa.table(kwargs, schema=ctx.input_type.arrow_schema).to_pylist()
        outputs = [
            self.process(
                RunContext(
                    node_id=ctx.node_id,
                    index=i,
                    rank=ctx.rank,
                    input_type=ctx.input_type,
                    output_type=ctx.output_type,
                ),
                **sample,
            )
            for i, sample in zip(ctx.index, batch, strict=True)
        ]

        # gather all outputs in case the process function
        # is a coroutine
        if self._is_process_async:
            outputs = await asyncio.gather(*outputs)

        return pa.array(outputs, type=ctx.output_type.arrow_type)
