"""Provides base classes for data augmentation in a data flow graph.

This module defines the base classes for data augmentation tasks within
a data flow graph framework. Data augmenters are responsible for filtering or
generating new data samples from on existing ones.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator as ABCAsyncIterable
from collections.abc import Iterable as ABCIterable
from itertools import chain
from types import GenericAlias
from typing import (
    Any,
    AsyncIterable,
    Concatenate,
    Generic,
    Iterable,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
    _GenericAlias,
    get_args,
    get_origin,
    overload,
)

import pyarrow as pa
from typing_extensions import Self

from hyped.common.typing import TraceIndexList

from ..typing import Feature
from .base import BaseNode, BaseNodeConfig, NodeProtocol, RunContext

Params = ParamSpec("Params")
Return = TypeVar("Return", covariant=True)


class _ProcessFunctionProtocol(Protocol, Generic[Params, Return]):
    """Protocol for synchronous processing functions in data augmenters."""

    def process(self, *args: Params.args, **kwargs: Params.kwargs) -> Iterable[Return]:
        """Processes data samples and returns an iterable of results."""
        ...


class _AsyncProcessFunctionProtocol(Protocol, Generic[Params, Return]):
    """Protocol for asynchronous processing functions in data augmenters."""

    def process(self, *args: Params.args, **kwargs: Params.kwargs) -> AsyncIterable[Return]:
        """Processes data samples asynchronously and returns an async iterable of results."""
        ...


class BaseDataAugmenterConfig(BaseNodeConfig):
    """Base configuration class for data augmenters.

    This class serves as the base configuration for data augmenters,
    inheriting from :code:`BaseNodeConfig` to provide configuration
    functionality specifically for data augmentation tasks.
    """


C = TypeVar("C", bound=BaseDataAugmenterConfig)


class BaseDataAugmenter(BaseNode[C], ABC):
    """Base class for data augmenters in a data flow graph.

    This class represents a data augmenter node in a data flow graph. Data augmenters
    modify or generate new samples from existing ones, which can include filtering or
    creating new data points. Subclasses of :code:`BaseDataAugmenter` must implement
    either the :code:`process` or the :code:`batch_process` method to define how the
    augmentation is applied to the input data.
    """

    def __new__(
        cls: Union[
            _ProcessFunctionProtocol[Concatenate[Self, RunContext, Params], Return],
            _AsyncProcessFunctionProtocol[Concatenate[Self, RunContext, Params], Return],
        ],
        *args: Any,
        **kwargs: Any,
    ) -> NodeProtocol[Params, Return]:
        """Creates a new instance of a data augmenter node.

        This method is responsible for creating an instance of the augmenter node, which
        follows either a synchronous or asynchronous processing protocol based on its type.

        Args:
            *args (Any): Positional arguments for node initialization.
            **kwargs (Any): Keyword arguments for node initialization.

        Returns:
            NodeProtocol[Params, Return]: An instance conforming to the node protocol.
        """
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, config: None | C = None, **kwargs) -> None:
        """Initializes the data augmenter with optional configuration.

        Args:
            config (None | C): Optional configuration instance for the augmenter.
            **kwargs (Any): Additional arguments for the augmenter configuration.
        """
        super().__init__(config, **kwargs)
        self._is_process_async = inspect.isasyncgenfunction(self.process)

    @property
    def signature(self) -> inspect.Signature:
        """Get the signature of the :func:`process` method.

        Returns the signature of the :func:`process` method with the :code:`ctx` parameter
        removed, keeping only the feature inputs modeled in the data flow graph.

        Returns:
            inspect.Signature: The signature of the :func:`process` method excluding
            the :code:`ctx` parameter.

        Raises:
            TypeError: If the return type of :func:`process` is not an iterable.
        """
        signature = inspect.signature(self.process)
        # remove the ctx argument of the process function
        params = signature.parameters.values()
        params = [param for param in params if param.name != "ctx"]
        # make sure output type is an iterable
        if not (
            isinstance(signature.return_annotation, (GenericAlias, _GenericAlias))
            and (
                # TODO: doesn't work for AsyncIterable
                get_origin(signature.return_annotation)
                in {Iterable, AsyncIterable, ABCIterable, ABCAsyncIterable}
            )
        ):
            raise TypeError()
        # get the flat return annotation
        return_annotation = get_args(signature.return_annotation)[0]
        # return the signature containing the remaining parameters
        return signature.replace(parameters=params, return_annotation=return_annotation)

    @overload
    async def process(
        self, ctx: RunContext, *args: Feature, **kwargs: Feature
    ) -> AsyncIterable[Feature]:
        ...

    @abstractmethod
    def process(self, ctx: RunContext, *args: Feature, **kwargs: Feature) -> Iterable[Feature]:
        """Defines the augmentation logic to be applied to individual samples.

        This method should be overridden by subclasses to define the augmentation of a single
        data sample. It may either be synchronous or asynchronous, depending on the subclass.

        It also defines the interface of the node and must be implemented by subclasses. Even if
        the :func:`batch_process` method implements the primary processing logic, :func:`process`
        is still required to define the input and output features of the node.

        Args:
            ctx (RunContext): Context information for the data augmenter's execution.
            *args (Feature): Positional input arguments.
            **kwargs (Feature): Keyword arguments.

        Returns:
            Iterable[Feature]: An iterable of augmented output samples, which can be multiple
            samples per input sample.
        """
        ...

    async def batch_process(
        self, ctx: RunContext, **kwargs: pa.Array
    ) -> tuple[pa.Array, TraceIndexList]:
        """Processes a batch of inputs and returns the batch of outputs along with trace indices.

        This method applies the augmentation process to each sample in the batch and tracks
        the index of the source sample for each output. It returns the augmented batch and
        the indices indicating the source of each output sample.

        Args:
            ctx (RunContext): Context information for the data augmenter's execution.
            **kwargs (pa.Array): Input features matching the signature of the :func:`process`
                function.

        Returns:
            tuple[pa.Array, TraceIndexList]: The output tuple of the following entries:
                - :code:`pa.Array`: An array containing all augmented output samples.
                - TraceIndexList: A list of trace indices corresponding indicating the
                  index of the source sample in the input batch that generated the
                  output sample. Specifically the i-th output sample is generated
                  from the trace_index[i]-th input example.
        """
        # apply process function to each sample in the input batch
        batch = pa.table(kwargs, schema=ctx.input_type.arrow_schema).to_pylist()
        calls = [
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

        # collect all outputs
        if self._is_process_async:
            # collect from async generators
            outputs = []
            for call in calls:
                outputs.append([sample async for sample in call])
        else:
            # collect from sync generators
            outputs = list(map(list, calls))

        # build trace indices for each output sample
        trace_index = ([i] * len(out) for i, out in enumerate(outputs))

        # chain outputs
        outputs = list(chain.from_iterable(outputs))
        trace_index = list(chain.from_iterable(trace_index))

        return pa.array(outputs, type=ctx.output_type.arrow_type), trace_index
