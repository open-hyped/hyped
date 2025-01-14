"""Provides base classes for nodes in a data flow graph.

This module defines base classes for nodes in a data flow graph. It
includes a base configuration class (:class:`BaseNodeConfig`) and a
generic base class (:class:`BaseNode`) for defining nodes with
configurable input and output types.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from inspect import Signature
from itertools import chain
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Hashable,
    Iterable,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)
from uuid import UUID, uuid4

import pyarrow as pa

from ..abc import AbstractDataFlow, AbstractDataFlowGraphBuilder
from ..features.dtypes import DType, MappingType
from ..features.features import Feature as _Feature
from ..features.features import build_feature_from_reference
from ..features.reference import ConcreteReference
from ..registry.config import BaseConfig, BaseConfigurable
from ..typing import Feature, Index, IndexList, NodeId, Rank
from ..utils import map_recursive


@dataclass
class RunSession:
    """Run session for the node execution.

    A :class:`RunSession` instance is shared along all nodes of a data flow
    and lives for the whole execution of a data flow. It manages the event
    loop that runs the data flow execution and allows to store context
    information.
    """

    _session_id: UUID = field(default_factory=uuid4, init=False)
    """A unique session ID."""

    _loop: asyncio.BaseEventLoop = field(default_factory=asyncio.new_event_loop, init=False)
    """The asyncio loop instance that runs the execution."""

    _context: dict[Hashable, Any] = field(default_factory=dict, init=False)
    """A dictionary to store the contextual data."""

    @property
    def session_id(self) -> UUID:
        """The unique session ID.

        Returns:
            UUID: A unique identifier for the session.
        """
        return self._session_id

    def set_context(self, key: Hashable, value: Any) -> None:
        """Set a context value associated with a specific key.

        Args:
            key (Hashable): The key for the context entry.
            value (Any): The value to associate with the key.
        """
        self._context[key] = value

    def get_context(self, key: Hashable) -> Any:
        """Retrieve the context value associated with a specific key.

        Args:
            key (Hashable): The key for the context entry to retrieve.

        Returns:
            Any: The value associated with the key, or None if the key is not found.
        """
        return self._context.get(key)


@dataclass(frozen=True)
class RunContext:
    """Context information for the node execution.

    Serves as an identifier for the specific call to the node within the data flow graph.
    This is particularly useful when a single node class is used multiple times in a data
    flow, as :class:`RunContext` identifies the specific instance of the node call, i.e.,
    the specific node in the flow graph.
    """

    session: None | RunSession
    """The run session instance.

    This attribute
    """

    node_id: NodeId
    """The id of the node in the data flow graph.

    This attribute serves as a unique identifier for the context, ensuring each processor
    call can be distinctly recognized within the flow graph.
    """

    index: Index | IndexList
    """The index or list of indices associated with the processor execution.

    This attribute is used to track the position or set of positions for processing data
    within a specific processor call. It could represent a single index or a list of indices,
    depending on how the data is partitioned or processed.
    """

    rank: Rank
    """The rank or position of the processor in a parallelized execution.

    This attribute identifies the specific rank or position of the processor in a parallelized
    system (e.g., in distributed or multi-threaded processing). The rank determines the processor's
    order or responsibility for a portion of the data during execution.
    """

    input_dtype: MappingType
    """The expected input data type for the processor.

    This attribute defines the type of the data that the processor is designed to handle as input.
    It typically maps the input data's structure or schema, providing context for how the data
    should be processed.
    """

    output_dtype: DType
    """The type of data the processor will produce as output.

    This attribute defines the expected structure or type of the output that the processor will
    generate. It provides information about the transformation or processing that the input data
    undergoes and the format of the resulting data.
    """

    def __hash__(self) -> int:
        """Returns a hash value based on the node ID.

        Returns:
            int: The hash value of the node ID.
        """
        return hash(self.node_id)  # pragma: not covered


P = ParamSpec("P")
R = TypeVar("R")


Backend: TypeAlias = Literal["python", "arrow"]


@dataclass(eq=True, frozen=True)
class ProcessMode:
    """Represents a processing mode for converting data to different formats."""

    batched: bool
    """Indicates whether the processing mode operates on batched data."""

    backend: Backend
    """Specifies the backend used for processing."""

    from_arrow_converters: ClassVar[dict[ProcessMode, Callable]] = {}
    """From arrow converter registry

    A registry of converter functions for transforming data from :class:`PyArrow` to the
    specified processing mode.
    """

    to_arrow_converters: ClassVar[dict[ProcessMode, Callable]] = {}
    """To arrow converter registry.

    A registry of converter functions for transforming data to :class:`PyArrow` from the
    specified processing mode.
    """

    def decorate(self, fn: Callable[P, R]) -> Callable[P, R]:
        """Decorates a function by associating it with this :class:`ProcessMode`.

        Args:
            fn (Callable[P, R]): The function to decorate.

        Returns:
            Callable[P, R]: The decorated function with the process mode attribute set.
        """
        fn.__hyped_process_mode__ = self
        return fn

    def set_default(self, fn: Callable[P, R]) -> Callable[P, R]:
        """Sets this :class:`ProcessMode` as the default for a function.

        Args:
            fn (Callable[P, R]): The function to set the default process mode for.

        Returns:
            Callable[P, R]: The function with the default process mode applied.
        """
        if not hasattr(fn, "__hyped_process_mode__"):
            return self.decorate(fn)

        return fn

    @classmethod
    def from_decorated_fn(cls, fn: Callable[P, R]) -> ProcessMode:
        """Retrieves the :class:`ProcessMode` associated with a decorated function.

        Args:
            fn (Callable[P, R]): The function from which to retrieve the process mode.

        Returns:
            ProcessMode: The associated process mode.

        Raises:
            RuntimeError: If the function does not have an associated process mode.
        """
        if not hasattr(fn, "__hyped_process_mode__"):
            raise RuntimeError("Function is not decorated with a 'ProcessMode'.")

        return fn.__hyped_process_mode__

    F = TypeVar("F", bound=Callable[P, R])

    @classmethod
    def register_from_arrow_converter(cls, mode: ProcessMode) -> Callable[[F], F]:
        """Decorator to register a function as a *from-Arrow* converter.

        Args:
            mode (ProcessMode): The process mode to register the converter for.

        Returns:
            Any: A callable that registers the function as a converter.
        """

        def decorator(fn: Callable) -> Callable:
            cls.from_arrow_converters[mode] = fn
            return fn

        return decorator

    @classmethod
    def register_to_arrow_converter(cls, mode: ProcessMode) -> Callable[[F], F]:
        """Decorator to register a function as a *to-Arrow* converter.

        Args:
            mode (ProcessMode): The process mode to register the converter for.

        Returns:
            Any: A callable that registers the function as a converter.
        """

        def decorator(fn: Callable) -> Callable:
            cls.to_arrow_converters[mode] = fn
            return fn

        return decorator

    def validate(self) -> ProcessMode:
        """Validates that the :class:`ProcessMode` is fully supported.

        Returns:
            ProcessMode: The validated process mode.

        Raises:
            Exception: If the ProcessMode is not supported (missing converters).
        """
        if not (
            (self in ProcessMode.from_arrow_converters)
            and (self in ProcessMode.to_arrow_converters)
        ):
            raise NotImplementedError(f"{self} is not fully supported.")

        return self

    def prepare(
        self, ctx: RunContext, **kwargs: pa.Array
    ) -> Iterable[tuple[RunContext, dict[str, Any]]]:
        """Prepares the data for processing in the current mode.

        Converts inputs to the required format using the registered from-Arrow converter.

        Args:
            ctx (RunContext): The context describing the current processing state.
            **kwargs (pa.Array): Input data arrays to convert.

        Yields:
            Iterable[tuple[RunContext, dict[str, Any]]]: A sequence of contexts and converted data.
        """
        # get the input converter function to the process mode
        converter = ProcessMode.from_arrow_converters[self]

        if self.batched:
            # apply the converter to the inputs
            yield ctx, converter(ctx.input_dtype.arrow_schema, **kwargs)

        else:
            # apply the converter to the inputs and yield samples with corresponding run contexts
            samples = converter(ctx.input_dtype.arrow_schema, **kwargs)
            yield from (
                (replace(ctx, index=i), sample)
                for i, sample in zip(ctx.index, samples, strict=True)
            )

    def finalize(self, ctx: RunContext, outputs: Iterable[Any]) -> pa.Array:
        """Finalizes the data after processing by converting outputs to Arrow.

        Args:
            ctx (RunContext): The context describing the current processing state.
                This argument expects the original context, not the prepared context
                generated by :code:`ProcessMode.prepare`.
            outputs (Iterable[Any]): The processed outputs to convert.

        Returns:
            pa.Array: The final Arrow array representation of the outputs.
        """
        # apply the converter function
        converter = ProcessMode.to_arrow_converters[self]
        return converter(ctx.output_dtype.arrow_type, outputs)


@ProcessMode.register_from_arrow_converter(ProcessMode(batched=False, backend="python"))
def _arrow_to_python_samples(schema: pa.Schema, **arrays: pa.Array) -> Iterable[dict[str, Any]]:
    """Converts Arrow arrays to an iterable of Python dictionaries."""
    return pa.table(arrays, schema=schema).to_pylist()


@ProcessMode.register_from_arrow_converter(ProcessMode(batched=True, backend="python"))
def _arrow_to_python_batch(schema: pa.Schema, **arrays: pa.Array) -> dict[str, list[Any]]:
    """Converts Arrow arrays to a Python dictionary of lists (batched mode)."""
    return pa.table(arrays, schema=schema).to_pydict()


@ProcessMode.register_from_arrow_converter(ProcessMode(batched=True, backend="arrow"))
def _arrow_to_arrow_batch(schema: pa.Schema, **arrays: pa.Array) -> dict[str, pa.Array]:
    """Passes through Arrow arrays as a dictionary (batched mode)."""
    return arrays


@ProcessMode.register_to_arrow_converter(ProcessMode(batched=False, backend="python"))
def _python_samples_to_arrow(arrow_type: pa.DataType, values: Iterable[Any]) -> pa.Array:
    """Converts an iterable of Python values to a single Arrow array."""
    return pa.array(values, type=arrow_type)


@ProcessMode.register_to_arrow_converter(ProcessMode(batched=True, backend="python"))
def _python_batch_to_arrow(arrow_type: pa.DataType, values: Iterable[list[Any]]) -> pa.Array:
    """Converts a batched iterable of Python lists to a single Arrow array."""
    return pa.array(chain.from_iterable(values), type=arrow_type)


@ProcessMode.register_to_arrow_converter(ProcessMode(batched=True, backend="arrow"))
def arrow_batch_to_arrow(arrow_type: pa.DataType, values: Iterable[pa.Array]) -> pa.Array:
    """Converts a batched iterable of Arrow arrays to a chunked Arrow array."""
    return pa.chunked_array(values, type=arrow_type)


F = TypeVar("F", bound=Callable[P, R])


def process_mode(batched: bool = False, backend: Backend = "python") -> Callable[[F], F]:
    """Decorator to specify the processing mode of a data processing function.

    This decorator associates a function with a :class:`ProcessMode`, specifying how the
    function handles its inputs and outputs during execution. The mode defines whether the
    function operates in batched or non-batched mode and which backend is used for processing
    (e.g., "python" or "arrow").

    Use the :code:`@process_mode(...)` decorator to annotate methods or functions that perform
    data processing. The decorator ensures the function is tagged with the appropriate
    processing mode, which can be validated or used during runtime.

    Args:
        batched (bool): Indicates if the function processes data in batches.
        backend (Backend): Specifies the backend used for processing.

    Returns:
        Callable[[F], F]: The decorator function that associates a function with the
        specified :class:`ProcessMode` instance.
    """
    return ProcessMode(batched=batched, backend=backend).validate().decorate


Params = ParamSpec("Params")
Return = TypeVar("Return", covariant=True)


class NodeProtocol(Protocol, Generic[Params, Return]):
    """Protocol for node-like objects in a data flow graph.

    This protocol defines the interface for nodes within a data flow graph, including methods
    for calling nodes with or without an explicit data flow. The :class:`NodeProtocol` is
    parameterized by :code:`Params`, defining the arguments, and :code:`Return`, defining the
    return type.
    """

    @overload
    def call(self, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        ...

    @overload
    def call(self, flow: AbstractDataFlow, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        ...

    def call(self, *args: Params.args, **kwargs: Params.kwargs) -> Return:  # noqa: D102
        ...  # pragma: not covered


class BaseNodeConfig(BaseConfig):
    """Base configuration class for nodes in a data flow graph."""


C = TypeVar("C", bound=BaseNodeConfig)


class BaseNode(BaseConfigurable[C], ABC):
    """Base class for nodes in a data flow graph.

    This class serves as a base for defining nodes in a data flow graph. Nodes are the building
    blocks of the graph, where each node represents a processing unit or transformation in the
    flow. It provides methods for configuring the node's signature, and interacting with the
    underlying graph structure.
    """

    def get_state(self) -> dict[str, Any]:
        """Get the node state.

        Returns:
            dict[str, Any]: The node's state, including :code:`__dict__`
            and :code:`__slots__` (if present).
        """
        state = {"__dict__": self.__dict__}
        if hasattr(self, "__slots__"):
            state["__slots__"] = self.__slots__
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Set the node state.

        Args:
            state (dict[str, Any]): A dictionary representing the state to restore,
                containing :code:`__dict__` and :code:`__slots__` (if present).
        """
        self.__dict__ = state["__dict__"]
        if hasattr(self, "__slots__"):
            self.__slots__ = state["__slots__"]

    @contextmanager
    def with_state(self, state: dict[str, Any]) -> Iterable[None]:
        """A context manager to temporarily set the node's state.

        Args:
            state (dict[str, Any]): A dictionary representing the state to temporarily set.
        """
        # capture the original state and overwrite with
        original_state = self.get_state()
        self.set_state(state)
        yield
        # reset to original state
        self.set_state(original_state)

    def initialize(self, ctx: RunContext) -> None:
        """Initialize the node for execution.

        This method is called before the data flow starts executing. It prepares
        the node for the upcoming execution by setting up any necessary state.
        Keep in mind that the node's state is reset after each execution,
        and this method is called again for each new run.

        The objects created in this method are managed by the :class:`RunSession`
        instance, which ensures that objects that cannot be pickled can still be
        used effectively in multiprocessing settings. In such settings, this method
        is executed in the child processes.

        Args:
            ctx (RunContext): The context of the current run.
        """
        ...

    @classmethod
    @property
    def Config(cls) -> type[C]:  # noqa: N802
        """Get the configuration type of the node.

        This property returns the configuration type associated with the node, which defines
        how the node is configured. It allows for accessing the specific configuration class
        for the node dynamically.

        Returns:
            type[C]: The configuration class type for the node.
        """
        return cls.config_type  # pragma: not covered

    @property
    @abstractmethod
    def signature(self) -> Signature:
        """Abstract method to define the node's signature.

        The signature method must be implemented by subclasses to return the signature
        of the node. The signature typically describes the expected input and output types
        for the node, providing essential metadata for the graph processing.

        Returns:
            Signature: The signature of the node, including its inputs and outputs.
        """
        ...

    def __str__(self) -> str:
        """Returns the string representation of the node.

        Returns:
            str: The class name of the node instance.
        """
        return type(self).__name__

    def _extract_builder_from_args(
        self, args: tuple[AbstractDataFlow | Feature], kwargs: dict[str, AbstractDataFlow | Feature]
    ) -> tuple[AbstractDataFlowGraphBuilder, tuple[Feature], dict[str, Feature]]:
        """Extract the data flow graph builder from the arguments.

        This method attempts to extract the data flow graph builder from either the positional or
        keyword arguments passed to the node. The method searches for an :code:`AbstractDataFlow`
        or :code:`Feature` to infer the associated graph builder from. If no flow is found, a
        runtime error is raised.

        Args:
            args (tuple[AbstractDataFlow | Feature]): Positional arguments that might contain
                the data flow or references to features.
            kwargs (dict[str, AbstractDataFlow | Feature]): Keyword arguments that might contain
                the data flow or references to features.

        Returns:
            tuple[AbstractDataFlowGraphBuilder, tuple[Feature], dict[str, Feature]]:
                A tuple containing the data flow graph builder and the remaining positional
                and keyword arguments.

        Raises:
            RuntimeError: If the builder cannot be inferred from the arguments.
        """
        # try to extract the data flow from
        # the positional arguments
        flow: None | AbstractDataFlow = (
            None
            if len(args) == 0
            else None
            if not isinstance(args[0], AbstractDataFlow)
            else args[0]
        )

        if flow is not None:
            # flow was found
            return flow._builder, args[1:], kwargs

        # try to extract the flow from the keyword arguments
        if "flow" in kwargs.keys():
            flow: AbstractDataFlow = kwargs.pop("flow")
            return flow._builder, args, kwargs

        # try to infer the flow from any feature argument
        all_args = chain(args, kwargs.values())
        features = filter(lambda f: isinstance(f, _Feature), all_args)

        # try to get the first reference in the arguments
        first: None | _Feature = next(features, None)
        if first is None:
            raise RuntimeError("DataFlow instance cannot be inferred from arguments!")

        # get the flow from the reference
        assert isinstance(first.ref, ConcreteReference)
        return first.ref._builder, args, kwargs

    @overload
    def call(self, *args: Feature, **kwargs: Feature) -> Feature:
        ...

    @overload
    def call(self, flow: AbstractDataFlow, *args: Feature, **kwargs: Feature) -> Feature:
        ...

    def call(
        self,
        *args: AbstractDataFlow | Feature,
        **kwargs: AbstractDataFlow | Feature,
    ) -> Feature:
        """Call the node, adding it to the underlying data flow.

        This method is adds the node in the context of a data flow graph. It validates the node's
        signature, processes the input arguments, and adds necessary constants and processor nodes
        to the graph. Finally, it returns the processed output feature.

        Args:
            *args (AbstractDataFlow | Feature): Positional arguments.
            **kwargs (AbstractDataFlow | Feature): Keyword arguments.

        Returns:
            _Feature: The feature resulting from the node's processing in the graph.

        Raises:
            RuntimeError: If the flow cannot be inferred from the arguments.
        """
        # extract the graph builder and prepare the input arguments
        builder, args, kwargs = self._extract_builder_from_args(args, kwargs)

        # bind arguments to signature and unpack dynamic keyword arguments
        bound_args = self.signature.bind(*args, **kwargs).arguments
        for param in self.signature.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                bound_args.update(bound_args.pop(param.name, {}))
                break

        # extract reference instances from all features in the inputs
        inputs = map_recursive(lambda _, x: x.ref if isinstance(x, _Feature) else x, bound_args)
        # add the compute node to the graph and build the output feature instance
        ref = builder.compute_node(self, inputs)
        return build_feature_from_reference(ref)
