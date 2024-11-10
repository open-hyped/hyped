"""Provides base classes for data aggregators in a data flow graph.

This module defines the base class for data aggregators, which manage the aggregation
of data within a data flow graph. It includes generic classes for defining data aggregators
with configurable input types and aggregation logic.

Classes:
    - :class:`DataAggregationManager`: Manager for handling data aggregation operations.
    - :class:`BaseDataAggregatorConfig`: Base class for data aggregator configurations.
    - :class:`BaseDataAggregator`: Base class for data aggregators.
"""
from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    Tuple,
    TypeVar,
    get_args,
    get_origin,
)

import pyarrow as pa
from typing_extensions import Self

from hyped.common._worker import manager as _manager  # noqa: F401

from ..typing import Feature
from .base import BaseNode, BaseNodeConfig, NodeProtocol, RunContext


class DataAggregationManager(object):
    """Manager for handling data aggregation operations.

    This class manages data aggregators, their thread-safe buffers, and synchronization
    locks to ensure safe concurrent updates during the data processing.

    Attributes:
        _value_buffer (dict): Thread-safe buffer for aggregation values.
        _state_buffer (dict): Thread-safe buffer for aggregation states.
        _locks (dict): Locks for synchronizing access to aggregators.
    """

    def __init__(
        self,
        aggregators: list[BaseDataAggregator],
        run_contexts: list[RunContext],
    ) -> None:
        """Initialize the DataAggregationManager.

        Args:
            aggregators (dict[str, BaseDataAggregator]): A list of aggregators.
            io_contexts (list[IOContext]): A list of contexts correspoding to the aggregators.
                Only used for call to :func:`initialize` function of each aggregator instance.
        """
        global _manager
        # create buffers
        value_buffer = {}
        state_buffer = {}
        # fill buffers with initial values from aggregators
        for agg, ctx in zip(aggregators, run_contexts):
            (
                value_buffer[ctx.node_id],
                state_buffer[ctx.node_id],
            ) = agg.initialize(ctx)
        assert all(isinstance(val, pa.Scalar) for val in value_buffer.values())
        # create thread-safe buffers
        self._value_buffer = _manager.dict(value_buffer)
        self._state_buffer = _manager.dict(state_buffer)
        # create a lock for each entry to synchronize access
        self._locks = {ctx.node_id: _manager.Lock() for ctx in run_contexts}
        self._locks = _manager.dict(self._locks)

    @property
    def values_proxy(self) -> MappingProxyType[str, Any]:
        """Get a read-only view of the aggregation values.

        Returns:
            MappingProxyType[str, Any]: A read-only view of the aggregation values.
        """
        return MappingProxyType(self._value_buffer)

    async def _safe_update(
        self, ctx: RunContext, aggregator: BaseDataAggregator, extracted: Any
    ) -> None:
        """Safely update an aggregation value.

        Args:
            ctx (RunContext): The io context object indicating the specific node to execute.
            aggregator (BaseDataAggregator): The aggregator object.
            extracted (Any): The values extracted from the input batch.
        """
        assert ctx.node_id in self._value_buffer
        # get the running event loop
        loop = asyncio.get_running_loop()
        # acquire the lock for the current aggregator
        await loop.run_in_executor(None, self._locks[ctx.node_id].acquire)
        # get current value and context
        value = self._value_buffer[ctx.node_id]
        state = self._state_buffer[ctx.node_id]
        # compute udpated value and context
        value, state = await aggregator.update(ctx, value, state, extracted)
        assert isinstance(value, pa.Scalar)
        # write new values to buffers
        self._value_buffer[ctx.node_id] = value
        self._state_buffer[ctx.node_id] = state
        # release lock
        self._locks[ctx.node_id].release()

    async def aggregate(
        self, aggregator: BaseDataAggregator, ctx: RunContext, inputs: dict[str, pa.Array]
    ) -> None:
        """Perform aggregation for a batch of inputs.

        Args:
            aggregator (BaseDataAggregator): The aggregator object.
            inputs (Batch): The batch of input samples.
            index (IndexList): The indices associated with the input samples.
            rank (Rank): The rank of the processor in a distributed setting.
            ctx (RunContext): Context information for the aggregator execution.
        """
        # extract values required for update from current input batch
        # and update the aggregated value and state
        extracted = await aggregator.extract(ctx, **inputs)
        await self._safe_update(ctx, aggregator, extracted)


Params = ParamSpec("Params")
Return = TypeVar("Return")


class _AggregatorProtocol(Protocol, Generic[Params, Return]):
    def extract(self, *args: Params.args, **kwargs: Params.kwargs) -> Any:
        ...

    def update(self, *args: Any, **kwargs: Any) -> tuple[Return, Any]:
        ...


class BaseDataAggregatorConfig(BaseNodeConfig):
    """Base configuration class for data aggregators.

    This class serves as the base configuration class for data aggregators.
    It inherits from :class:`BaseConfig`, providing basic configuration
    functionality for data aggregation tasks.
    """


C = TypeVar("C", bound=BaseDataAggregatorConfig)


class BaseDataAggregator(BaseNode[C], ABC):
    """Base class for data aggregators.

    This class serves as the base for all data aggregators, defining the necessary
    interfaces and methods for implementing custom aggregators.
    """

    def __new__(
        cls: (_AggregatorProtocol[Concatenate[Self, RunContext, Params], Return]),
        *args: Any,
        **kwargs: Any,
    ) -> NodeProtocol[Params, Return]:
        return super().__new__(cls, *args, **kwargs)

    @property
    def signature(self) -> inspect.Signature:
        param_annotations = inspect.signature(self.extract).parameters
        return_annotation = inspect.signature(self.update).return_annotation

        # check return type annotation of update function
        if (get_origin(return_annotation) not in {tuple, Tuple}) or (
            len(get_args(return_annotation)) != 2
        ):
            raise TypeError("Return Annotation", return_annotation)

        # remove the ctx argument of the process function
        param_annotations = param_annotations.values()
        param_annotations = [param for param in param_annotations if param.name != "ctx"]
        # get the aggregation value feature type from the return annotation
        return_annotation = get_args(return_annotation)[0]

        # build the signature
        return inspect.Signature(parameters=param_annotations, return_annotation=return_annotation)

    Value = TypeVar("Value", bound=Feature)
    State = TypeVar("State")
    Extracted = TypeVar("Extract")

    @abstractmethod
    def initialize(self, ctx: RunContext) -> tuple[Value, State]:
        """Initialize the aggregator with the given features.

        Args:
            io (IOContext): The execution context object wrapping the
                input and output features.

        Returns:
            tuple[Value, State]: The initial value and state for the aggregator.
        """
        ...

    # TODO: currently the node signature input arguments are read from the extract function
    #       which requires them to be annotated with feature types, however the input values
    #       are pyarrow arrays and not scalars
    #       A feature annotation is equivalent with a scalar / primitive but not with an array
    #       Should we say a feature can be a scalar or an array? The specific type must then be
    #       inferred from context. extract -> arrays, update -> scalar (note that the return
    #       type of the update function is a scalar but is also annotated as a feature)
    @abstractmethod
    async def extract(self, ctx: RunContext, *args: Feature, **kwargs: Feature) -> Extracted:
        """Extract necessary values from the inputs for aggregation.

        Args:
            ctx (RunContext): The run context object.
            *args (Feature): Positional input arguments.
            **kwargs (Feature): Keyword input arguments.

        Returns:
            Extract: The extracted context values required for aggregation.
        """
        ...

    @abstractmethod
    async def update(
        self, ctx: RunContext, val: Value, state: State, extracted: Extracted
    ) -> tuple[Value, State]:
        """Update the aggregation value and context.

        Args:
            ctx (RunContext): The run context object.
            val (Value): The current aggregation value.
            state (State): The current aggregation state.
            extracted (Extract): The values extracted from the input batch.

        Returns:
            tuple[Value, State]: The updated aggregation value and state.
        """
        ...
