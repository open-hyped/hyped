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
from dataclasses import replace
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
    runtime_checkable,
)

import pyarrow as pa
from typing_extensions import Self

from hyped.common._worker import manager as _manager  # noqa: F401

from ..features.dtypes import MappingType
from ..typing import Feature
from .base import BaseNode, BaseNodeConfig, NodeProtocol, ProcessMode, RunContext


class DataAggregationManager(object):
    """Manager for handling data aggregation operations.

    This class manages data aggregators, their thread-safe buffers, and synchronization
    locks to ensure safe concurrent updates during the data processing.
    """

    def __init__(
        self,
        aggregators: list[BaseDataAggregator],
        run_contexts: list[RunContext],
    ) -> None:
        """Initialize the DataAggregationManager.

        Args:
            aggregators (dict[str, BaseDataAggregator]): A list of aggregators.
            run_contexts (list[RunContext]): A list of contexts correspoding to the aggregators.
                Only used for call to :func:`initialize` function of each aggregator instance.
        """
        global _manager
        # create buffers
        value_buffer = {}
        state_buffer = {}
        # fill buffers with initial values from aggregators
        for agg, ctx in zip(aggregators, run_contexts, strict=True):
            val, state = agg.initialize(ctx)
            val = pa.array([val], type=ctx.output_type.arrow_type)
            # write values to buffers
            value_buffer[ctx.node_id] = val
            state_buffer[ctx.node_id] = state
        # create thread-safe buffers
        self._value_buffer = _manager.dict(value_buffer)
        self._state_buffer = _manager.dict(state_buffer)
        # create a lock for each entry to synchronize access
        self._locks = {ctx.node_id: _manager.Lock() for ctx in run_contexts}
        self._locks = _manager.dict(self._locks)

    @property
    def values_proxy(self) -> MappingProxyType[str, pa.Array]:
        """Get a read-only view of the aggregation values.

        Returns:
            MappingProxyType[str, pa.Array]: A read-only view of the aggregation values.
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

        # get the process mode of the update function
        mode = ProcessMode.from_decorated_fn(aggregator.update)

        # get the running event loop
        loop = asyncio.get_running_loop()
        # acquire the lock for the current aggregator
        await loop.run_in_executor(None, self._locks[ctx.node_id].acquire)
        # get current value and context
        value = self._value_buffer[ctx.node_id]
        state = self._state_buffer[ctx.node_id]

        # prepare the value for the update
        value_type = MappingType.construct({"value": ctx.output_type})
        inputs = mode.prepare(replace(ctx, input_type=value_type), value=value)
        # should only contain a single input tuple
        update_ctx, update_kw = next(iter(inputs))
        update_ctx = replace(update_ctx, input_type=ctx.input_type)
        # run the aggregator update function
        value, state = await aggregator.update(update_ctx, update_kw["value"], state, extracted)

        # write new values to buffers
        self._value_buffer[ctx.node_id] = mode.finalize(ctx, [value])
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
        # get the process mode for the extract function
        mode = ProcessMode.from_decorated_fn(aggregator.extract)

        for c, kw in mode.prepare(ctx, **inputs):
            # extract values required for update from current input batch
            # and update the aggregated value and state
            extracted = await aggregator.extract(c, **kw)
            await self._safe_update(c, aggregator, extracted)


Params = ParamSpec("Params")
Return = TypeVar("Return")


@runtime_checkable
class _AggregatorProtocol(Protocol, Generic[Params, Return]):
    async def extract(self, *args: Params.args, **kwargs: Params.kwargs) -> Any:
        ...  # pragma: not covered

    async def update(self, *args: Any, **kwargs: Any) -> tuple[Return, Any]:
        ...  # pragma: not covered


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
    interfaces and methods for implementing custom aggregators. Subclasses must
    implement the `extract` and `update` methods, which define the logic for
    retrieving and updating aggregated values in the data flow graph.
    """

    def __new__(
        cls: (_AggregatorProtocol[Concatenate[Self, RunContext, Params], Return]),
        *args: Any,
        **kwargs: Any,
    ) -> NodeProtocol[Params, Return]:
        """Creates a new instance of a data aggregator node.

        Args:
            *args (Any): Positional arguments for node initialization.
            **kwargs (Any): Keyword arguments for node initialization.

        Returns:
            NodeProtocol[Params, Return]: An instance conforming to the node protocol.
        """
        return super().__new__(cls)

    @classmethod
    def _check_signature(cls) -> bool:
        """Validate the signatures of the :code:`extract` and :code:`update` methods.

        This method ensures that the :code:`extract` and :code:`update` methods conform to
        the expected protocols and signature requirements.

        Validations:
            1. The class must implement the :class:`_AggregatorProtocol`.
            2. The :code:`extract` method must have the :code:`ctx` argument as
               the second parameter (after :code:`self`) annotated with :class:`RunContext`.
            3. The `update` method must return a tuple containing two elements: the new
               aggregation value and the state.

        Returns:
            bool: :code:`True` if the signatures match the expected protocols,
            :code:`False` otherwise.
        """
        # check the protocol, in practice this can never fire because
        # the process function is abstract
        if not isinstance(cls, _AggregatorProtocol):
            return False  # pragma: not covered

        extract_signature = inspect.signature(cls.extract)
        # check the context argument is the first argument
        # after self and has the correct annotation
        if not (
            ("ctx" in extract_signature.parameters)
            and (list(extract_signature.parameters.keys()).index("ctx") == 1)
            and (extract_signature.parameters["ctx"].annotation is RunContext)
        ):
            return False

        update_signature = inspect.signature(cls.update)
        # return of update function must be a tuple containing the
        # new aggregation value and state
        return (
            get_origin(update_signature.return_annotation) in (tuple, Tuple)
            and len(get_args(update_signature.return_annotation)) == 2
        )

    @classmethod
    def __init_subclass__(cls) -> None:
        """Hook method to validate the subclass during initialization.

        This method ensures that any subclass of :class:`BaseDataAggregator` adheres
        to the required :code:`extract` and :code:`update` method signatures and sets default
        :class:`ProcessMode` configurations for these methods.

        Workflow:
            1. Sets the default :class:`ProcessMode` for the :code:`extract` method to batched
               mode.
            2. Sets the default :class:`ProcessMode` for the :code:`update` method to non-batched
               mode.
            3. Calls :code:`_check_signature` to validate the method signatures.
            4. Raises a :class:`TypeError` if the subclass does not conform to the expected
               signatures and protocols.

        Raises:
            TypeError: If the subclass does not implement valid :code:`extract` and :code:`update`
                methods as per the :class:`_AggregatorProtocol`.
        """
        # set default process modes for extract and update functions
        ProcessMode(batched=True, backend="python").validate().set_default(cls.extract)
        ProcessMode(batched=False, backend="python").validate().set_default(cls.update)

        if not cls._check_signature():
            raise TypeError(
                f"The class '{cls.__name__}' must implement valid 'extract' and 'update' "
                "methods conforming to the '_AggregatorProtocol'. Ensure that 'extract' "
                "has a correctly annotated 'ctx' argument and that 'update' returns a "
                "(value, state)-tuple."
            )

    @property
    def signature(self) -> inspect.Signature:
        """Retrieve the signature for the aggregator node.

        This method constructs a signature for the aggregator node by combining
        the parameters of the :func:`extract` method with the return annotation
        of the :func:`update` method. The :code:`ctx` parameter is excluded
        from the parameter list, ensuring that the signature reflects only
        the feature inputs relevant to the aggregation process.

        The return annotation of the signature is derived from the :func:`update` method,
        representing the aggregated feature type produced by the aggregator node.

        Returns:
            inspect.Signature: A constructed signature for the aggregator node,
            with parameters from :func:`extract` (excluding :code:`ctx`) and a
            return annotation based on the feature type from :func:`update`.

        Raises:
            TypeError: If the return annotation of :func:`update` does not represent
            a valid aggregated feature type.
        """
        param_annotations = inspect.signature(self.extract).parameters
        return_annotation = inspect.signature(self.update).return_annotation
        # remove the ctx argument of the process function
        param_annotations = param_annotations.values()
        param_annotations = [param for param in param_annotations if param.name != "ctx"]
        # get the aggregation value feature type from the return annotation
        # note that the return annotation is validated to be a two-tuple in
        # the init-subclass methid
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
            ctx (RunContext): The run context object.

        Returns:
            tuple[Value, State]: The initial value and state for the aggregator.
        """
        ...

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
