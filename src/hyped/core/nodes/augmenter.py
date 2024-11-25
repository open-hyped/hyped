"""Provides base classes for data augmentation in a data flow graph.

This module defines the base classes for data augmentation tasks within
a data flow graph framework. Data augmenters are responsible for filtering or
generating new data samples from on existing ones.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
    Any,
    AsyncIterable,
    Concatenate,
    Generic,
    Iterable,
    ParamSpec,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
    runtime_checkable,
)

import pyarrow as pa
from typing_extensions import Self

from ..typing import Feature, TraceIndexList
from .base import BaseNode, BaseNodeConfig, NodeProtocol, ProcessMode, RunContext

Params = ParamSpec("Params")
Return = TypeVar("Return", covariant=True)


@runtime_checkable
class _ProcessFunctionProtocol(Protocol, Generic[Params, Return]):
    """Protocol for synchronous processing functions in data augmenters."""

    def process(self, *args: Params.args, **kwargs: Params.kwargs) -> Iterable[Return]:
        """Processes data samples and returns an iterable of results."""
        ...  # pragma: not covered


@runtime_checkable
class _AsyncProcessFunctionProtocol(Protocol, Generic[Params, Return]):
    """Protocol for asynchronous processing functions in data augmenters."""

    def process(self, *args: Params.args, **kwargs: Params.kwargs) -> AsyncIterable[Return]:
        """Processes data samples asynchronously and returns an async iterable of results."""
        ...  # pragma: not covered


@runtime_checkable
class _BatchProcessFunctionProtocol(Protocol, Generic[Params, Return]):
    """Protocol for batched processing functions in data augmenters."""

    def process(self, *args: Params.args, **kwargs: Params.kwargs) -> tuple[Return, TraceIndexList]:
        """Processes data batches and returns the result and corresponding trace index list."""
        ...  # pragma: not covered


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
            _BatchProcessFunctionProtocol[Concatenate[Self, RunContext, Params], Return],
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

    @classmethod
    def _check_signature(cls) -> bool:
        """Validate the signature of the :code:`process`.

        This method ensures that the :code:`process` method conforms to one
        of the valid process function protocols. The signature is checked based on
        the specific requirements for synchronous, asynchronous, and batched processing
        modes.

        Validations:
            1. The :code:`process` method must implement one of the following protocols, i.e.
               :class:`_ProcessFunctionProtocol`, :class:`_AsyncProcessFunctionProtocol`
               :class:`_BatchProcessFunctionProtocol`
            2. The :code:`process` method must have a :code:`ctx` argument as the second
               parameter (immediately after :code:`self`) annotated with :class:`RunContext`.
            3. If the process mode is batched:
               - The method must conform to :class:`_BatchProcessFunctionProtocol`.
               - The return annotation must be a tuple where the second element is
                 :class:`TraceIndexList`.
            4. If the process mode is non-batched:
               - The method must conform to :class:`_ProcessFunctionProtocol` or
                 :class:`_AsyncProcessFunctionProtocol`.

        Returns:
            bool: :code:`True` if the :code:`process` method matches one of the valid protocols,
            :code:`False` otherwise.
        """
        # check the protocol, in practice this can never fire because
        # the process function is abstract
        if not isinstance(
            cls,
            (
                _ProcessFunctionProtocol,
                _AsyncProcessFunctionProtocol,
                _BatchProcessFunctionProtocol,
            ),
        ):
            return False  # pragma: not covered

        signature = inspect.signature(cls.process)
        # check the context argument is the first argument
        # after self and has the correct annotation
        if not (
            ("ctx" in signature.parameters)
            and (list(signature.parameters.keys()).index("ctx") == 1)
            and (signature.parameters["ctx"].annotation is RunContext)
        ):
            return False

        # get the assigned process mode
        mode = ProcessMode.from_decorated_fn(cls.process)

        if mode.batched:
            # make sure the process function matched the batch process protocol
            # and the return annotation is correct
            return (
                isinstance(cls, _BatchProcessFunctionProtocol)
                and get_origin(signature.return_annotation) in (tuple, Tuple)
                and len(get_args(signature.return_annotation)) == 2
                and get_args(signature.return_annotation)[1] == TraceIndexList
            )

        else:
            # make sure the process function matches any of the non-batched protocols
            return isinstance(cls, (_ProcessFunctionProtocol, _AsyncProcessFunctionProtocol))

    @classmethod
    def __init_subclass__(cls) -> None:
        """Hook method to validate the subclass during its initialization.

        This method ensures that any subclass of the node conforms to the required
        :code:`process` method signature and sets a default :class:`ProcessMode` if
        not explicitly defined. If the subclass does not meet the signature requirements,
        a :class:`TypeError` is raised. The workflow is as follows

        1. Applies a default :class:`ProcessMode` to the :code:`process` method if no
           explicit configuration is set.
        2. Calls :code:`_check_signature` to validate the :code:`process` method's
           arguments, annotations, and protocol conformity.
        3. Raises a :class:`TypeError` if the :code:`process` method does not match the
           expected signature.

        Raises:
            TypeError: If the subclass does not implement a valid :code:`process` method
            that adheres to one of the defined process function protocols.
        """
        # set default process mode for process function
        ProcessMode(batched=False, backend="python").validate().set_default(cls.process)

        if not cls._check_signature():
            raise TypeError(
                f"The class '{cls.__name__}' must implement a 'process' method that matches "
                "the signature a valid augmentation signature (i.e. '_ProcessFunctionProtocol', "
                "'_AsyncProcessFunctionProtocol' or '_BatchProcessFunctionProtocol'). Ensure the "
                "method's arguments and return type conform to one of the expected protocols."
            )

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

        # get the flat return annotation
        # this works for all supported protocols
        return_annotation = get_args(signature.return_annotation)[0]
        # return the signature containing the remaining parameters
        return signature.replace(parameters=params, return_annotation=return_annotation)

    @overload
    async def process(
        self, ctx: RunContext, *args: Feature, **kwargs: Feature
    ) -> AsyncIterable[Feature]:
        ...

    @overload
    def process(self, ctx: RunContext, *args: Feature, **kwargs: Feature) -> Iterable[Feature]:
        ...

    @overload
    def process(
        self, ctx: RunContext, *args: Feature, **kwargs: Feature
    ) -> tuple[Feature, TraceIndexList]:
        ...

    @overload
    async def process(
        self, ctx: RunContext, *args: Feature, **kwargs: Feature
    ) -> tuple[Feature, TraceIndexList]:
        ...

    @abstractmethod
    def process(
        self, ctx: RunContext, *args: Feature, **kwargs: Feature
    ) -> Iterable[Feature] | AsyncIterable[Feature] | tuple[Feature, TraceIndexList]:
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
            Union[Iterable[Feature], AsyncIterable[Feature], tuple[Feature, TraceIndexList]]:
                - **Iterable[Feature]**: If the process function is synchronous, it returns an
                  iterable of augmented output samples, potentially producing multiple outputs
                  per input.
                - **AsyncIterable[Feature]**: If the process function is asynchronous, it returns
                  an async iterable of augmented output samples, following the same logic as
                  the synchronous mode.
                - **tuple[Feature, TraceIndexList]**: If the process function operates in batched
                  mode, it returns:
                    - **Feature**: A batch of augmented output samples.
                    - **TraceIndexList**: A list of trace indices mapping each output sample to
                      the corresponding source sample in the input batch. Specifically, the
                      :code:`i`-th output sample originates from the :code:`trace_index[i]`-th input
                      example.
        """
        ...

    async def run(
        self, ctx: RunContext, arrays: dict[str, pa.Array]
    ) -> tuple[pa.Array, TraceIndexList]:
        """Execute the main processing logic for the data augmenter.

        This method serves as the primary entry point for processing data within
        a data flow graph, returning both the processed outputs and their associated
        trace indices. It orchestrates the execution of the :code:`process` method
        according to the configured :class:`ProcessMode`, handling input preparation,
        processing, and output finalization. In detail, the workflow is:

        1. Determine the processing mode (:class:`ProcessMode`) based on the
           :code:`process` method's configuration.
        2. Prepare the input data using the :class:`ProcessMode.prepare` method.
        3. Apply the :code:`process` method to the prepared inputs. If the method
           is asynchronous, the outputs are awaited using :code:`asyncio.gather`.
        4. Depending on whether the processing is batched:
           - If batched:
             a. Separate the outputs and their associated trace indices from the
                :code:`process` method's results.
             b. Concatenate the trace indices and finalize the outputs as a
                :code:`PyArrow` array.
           - If non-batched:
             a. Collect all outputs and trace indices from the :code:`process` method,
                consuming asynchronous or synchronous iterators as appropriate.
             b. Chain the outputs and finalize them as a :code:`PyArrow` array.
             c. Build a trace index list that maps each output sample back to its
                corresponding input.

        Args:
            ctx (RunContext): The execution context containing.
            arrays (dict[str, pa.Array]): A dictionary mapping input names to
                :code:`PyArrow` arrays, representing the input data to be processed.

        Returns:
            tuple[pa.Array, TraceIndexList]: A tuple containing the processed output
            as a :code:`PyArrow` array and a list of trace indices mapping the output
            samples to their respective input sources.
        """
        # get the process mode
        mode = ProcessMode.from_decorated_fn(self.process)
        # prepare inputs and apply process function to all inputs
        inputs = mode.prepare(ctx, **arrays)
        calls = (self.process(c, **kw) for c, kw in inputs)

        if mode.batched:
            # gather the call outputs
            if self._is_process_async:
                calls = await asyncio.gather(*calls)
            # separate outputs from trace indices
            outputs, trace_index = zip(*calls, strict=True)
            # finalize outputs and concatenate trace indices
            outputs = mode.finalize(ctx, outputs)
            trace_index = list(chain.from_iterable(trace_index))
            return outputs, trace_index

        else:
            # consume all iterables
            if self._is_process_async:
                outputs = []
                for call in calls:
                    outputs.append([x async for x in call])
            else:
                outputs = list(map(list, calls))

            # build trace indices for each output sample
            trace_index = ([i] * len(out) for i, out in enumerate(outputs))
            # chain outputs
            outputs = mode.finalize(ctx, chain.from_iterable(outputs))
            trace_index = list(chain.from_iterable(trace_index))
            return outputs, trace_index
