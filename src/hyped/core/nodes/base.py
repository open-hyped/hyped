"""Provides base classes for nodes in a data flow graph.

This module defines base classes for nodes in a data flow graph. It
includes a base configuration class (:class:`BaseNodeConfig`) and a
generic base class (:class:`BaseNode`) for defining nodes with
configurable input and output types.
"""
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import partial
from inspect import Signature
from itertools import chain
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

import pyarrow as pa

from hyped._registry.config import BaseConfig, BaseConfigurable

from ..abstract import AbstractDataFlow, AbstractDataFlowGraph
from ..features.engine import FeatureEngine
from ..features.features import _Feature
from ..features.types import MappingType, Type
from ..typing import Feature, Index, IndexList, NodeId, Rank


@dataclass(frozen=True)
class RunContext:
    """Context information for the data processors execution.

    Serves as an identifier for the specific call to the processor within the data flow graph.
    This is particularly useful when a single processor class is used multiple times in a data
    flow, as :class:`RunContext` identifies the specific instance of the processor call, i.e.,
    the specific node in the flow graph.
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

    input_type: MappingType
    """The expected input type for the processor.

    This attribute defines the type of the data that the processor is designed to handle as input.
    It typically maps the input data's structure or schema, providing context for how the data
    should be processed.
    """

    output_type: Type
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
        return hash(self.node_id)


P = ParamSpec("P")
R = TypeVar("R")


Backend: TypeAlias = Literal["python", "arrow"]


@dataclass(eq=True, frozen=True)
class ProcessMode:
    batched: bool

    backend: Backend

    from_arrow_converters: ClassVar[dict[ProcessMode, Callable]] = {}
    to_arrow_converters: ClassVar[dict[ProcessMode, Callable]] = {}

    def __call__(self, fn: Callable[P, R]) -> Callable[P, R]:
        # decorator
        fn.__hyped_process_mode__ = self
        return fn

    def set_default(self, fn: Callable[P, R]) -> Callable[P, R]:
        if not hasattr(fn, "__hyped_process_mode__"):
            return self(fn)

        return fn

    @classmethod
    def from_decorated_fn(cls, fn: Callable[P, R]) -> ProcessMode:
        if not hasattr(fn, "__hyped_process_mode__"):
            raise RuntimeError()

        return fn.__hyped_process_mode__

    @classmethod
    def register_from_arrow_converter(cls, mode: ProcessMode) -> Any:
        return partial(operator.setitem, cls.from_arrow_converters, mode)

    @classmethod
    def register_to_arrow_converter(cls, mode: ProcessMode) -> Any:
        return partial(operator.setitem, cls.to_arrow_converters, mode)

    def validate(self) -> ProcessMode:
        if not (
            (self in ProcessMode.from_arrow_converters)
            and (self in ProcessMode.to_arrow_converters)
        ):
            # TODO: mode not supported error message
            raise Exception()

        return self

    def prepare(
        self, ctx: RunContext, **kwargs: pa.Array
    ) -> Iterable[tuple[RunContext, dict[str, Any]]]:
        # get the input converter function to the process mode
        assert self in ProcessMode.from_arrow_converters
        converter = ProcessMode.from_arrow_converters[self]

        if self.batched:
            # apply the converter to the inputs
            yield ctx, converter(ctx.input_type.arrow_schema, **kwargs)

        else:
            # apply the converter to the inputs and yield samples with corresponding run contexts
            samples = converter(ctx.input_type.arrow_schema, **kwargs)
            yield from (
                (replace(ctx, index=i), sample)
                for i, sample in zip(ctx.index, samples, strict=True)
            )

    def finalize(self, ctx: RunContext, outputs: Iterable[Any]) -> pa.Array:
        # get the converter function
        assert self in ProcessMode.to_arrow_converters
        converter = ProcessMode.to_arrow_converters[self]
        # run the converter on the outputs
        return converter(ctx.output_type.arrow_type, outputs)


@ProcessMode.register_from_arrow_converter(ProcessMode(batched=False, backend="python"))
def _arrow_to_python_samples(schema: pa.Schema, **arrays: pa.Array) -> Iterable[dict[str, Any]]:
    return pa.table(arrays, schema=schema).to_pylist()


@ProcessMode.register_from_arrow_converter(ProcessMode(batched=True, backend="python"))
def _arrow_to_python_batch(schema: pa.Schema, **arrays: pa.Array) -> dict[str, list[Any]]:
    return pa.table(arrays, schema=schema).to_pydict()


@ProcessMode.register_from_arrow_converter(ProcessMode(batched=True, backend="arrow"))
def _arrow_to_arrow_batch(schema: pa.Schema, **arrays: pa.Array) -> dict[str, pa.Array]:
    return arrays


@ProcessMode.register_to_arrow_converter(ProcessMode(batched=False, backend="python"))
def _python_samples_to_arrow(arrow_type: pa.DataType, values: Iterable[Any]) -> pa.Array:
    return pa.array(values, type=arrow_type)


@ProcessMode.register_to_arrow_converter(ProcessMode(batched=True, backend="python"))
def _python_samples_to_arrow(arrow_type: pa.DataType, values: Iterable[list[Any]]) -> pa.Array:
    return pa.array(chain.from_iterable(values), type=arrow_type)


@ProcessMode.register_to_arrow_converter(ProcessMode(batched=True, backend="arrow"))
def _python_samples_to_arrow(arrow_type: pa.DataType, values: Iterable[pa.Array]) -> pa.Array:
    return pa.chunked_array(values, type=arrow_type)


F = TypeVar("F", bound=Callable[P, R])


def process_mode(batched: bool = False, backend: Backend = "python") -> Callable[[F], F]:
    return ProcessMode(batched=batched, backend=backend).validate()


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

    def call(self, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        """Call the node, adding it to the underlying data flow.

        This method is used to execute the node within the context of a data flow graph.
        It validates the node's signature, processes the input arguments, and adds necessary
        constants and processor nodes to the graph. Finally, it returns the processed output
        feature.

        Args:
            *args (Params.args): Positional arguments for the node, which may include features
                or data flows.
            **kwargs (Params.kwargs): Keyword arguments for the node, which may include features
                or data flows.

        Returns:
            Return: The feature resulting from the node's processing within the graph.

        Raises:
            RuntimeError: If the flow cannot be inferred from the arguments.
        """
        ...


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
        return cls.config_type

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

    def _extract_graph_from_args(
        self, args: tuple[AbstractDataFlow | Feature], kwargs: dict[str, AbstractDataFlow | Feature]
    ) -> tuple[AbstractDataFlowGraph, tuple[Feature], dict[str, Feature]]:
        """Extract the data flow graph from the arguments.

        This method attempts to extract the data flow graph from either the positional or
        keyword arguments passed to the node. The method searches for an :code:`AbstractDataFlow`
        or :code:`_Feature` to infer the associated graph. If a flow cannot be determined, a
        runtime error is raised.

        Args:
            args (tuple[AbstractDataFlow | Feature]): Positional arguments that might contain
                the data flow or references to features.
            kwargs (dict[str, AbstractDataFlow | Feature]): Keyword arguments that might contain
                the data flow or references to features.

        Returns:
            tuple[AbstractDataFlowGraph, tuple[Feature], dict[str, Feature]]:
                A tuple containing the data flow graph, remaining positional arguments, and
                keyword arguments with extracted features.

        Raises:
            RuntimeError: If the flow cannot be inferred from the arguments.
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
            return flow._graph, args[1:], kwargs

        # try to extract the flow from the keyword arguments
        if "flow" in kwargs.keys():
            flow: AbstractDataFlow = kwargs.pop("flow")
            return flow._graph, args, kwargs

        # try to infer the flow from any feature argument
        all_args = chain(args, kwargs.values())
        references = filter(lambda f: isinstance(f, _Feature), all_args)

        # try to get the first reference in the arguments
        reference: _Feature = next(references, None)

        if reference is None:
            raise RuntimeError("DataFlow instance cannot be inferred from arguments!")

        # get the flow from the reference
        return reference.ref._graph, args, kwargs

    @overload
    def call(self, *args: Feature, **kwargs: Feature) -> _Feature:
        ...

    @overload
    def call(self, flow: AbstractDataFlow, *args: Feature, **kwargs: Feature) -> _Feature:
        ...

    def call(
        self,
        *args: AbstractDataFlow | Feature,
        **kwargs: AbstractDataFlow | Feature,
    ) -> _Feature:
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
        # extract the data flow graph from the given arguments
        graph, args, kwargs = self._extract_graph_from_args(args, kwargs)

        # create the type engine from the node signature
        name = f"{type(self).__qualname__}.call"
        engine = FeatureEngine(name, self.config, self.signature)

        # validate the node signature and input arguments
        engine.validate_signature()
        engine.validate_arguments(*args, **kwargs)
        # split the input features from the input constants
        references, consts, const_dtypes = engine.get_references_and_consts(*args, **kwargs)

        # add all constants to the graph
        for key, val in consts.items():
            references[key] = graph.add_const_node(val, const_dtypes[key])

        # add the node and return the output feature
        ref = graph.add_processor_node(self, references)
        return graph.get_feature_from_reference(ref)
