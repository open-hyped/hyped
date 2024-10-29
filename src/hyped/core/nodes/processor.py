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
    Subclasses of `BaseDataProcessor` implement specific process functions that map input features
    to output features. Custom data processors must either override the `batch_process` method or
    the `process` method to define their processing logic.

    Attributes:
        _is_process_async (bool): A flag indicating whether the :class:`BaseDataProcessor.process`
            function is asynchronous.
    """

    def __new__(
        cls: _ProcessFunctionProtocol[Concatenate[Self, RunContext, Params], Return],
        *args: Any,
        **kwargs: Any,
    ) -> NodeProtocol[Params, Return]:
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
        ...

    async def batch_process(self, ctx: RunContext, **kwargs: pa.Array) -> pa.Array:
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
            for i, sample in zip(ctx.index, batch)
        ]

        # gather all outputs in case the process function
        # is a coroutine
        if self._is_process_async:
            outputs = await asyncio.gather(*outputs)

        return pa.array(outputs, type=ctx.output_type.arrow_type)
