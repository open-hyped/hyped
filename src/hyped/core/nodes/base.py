"""Provides base classes for nodes in a data flow graph.

This module defines base classes for nodes in a data flow graph. It
includes a base configuration class (:class:`BaseNodeConfig`) and a
generic base class (:class:`BaseNode`) for defining nodes with
configurable input and output types.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import Signature
from itertools import chain
from typing import Generic, ParamSpec, Protocol, TypeAlias, TypeVar, overload

from hyped._registry.config import BaseConfig, BaseConfigurable
from hyped.common.typing import Index, IndexList, NodeId, Rank

from ..abstract import AbstractDataFlow, AbstractDataFlowGraph
from ..features.engine import FeatureEngine
from ..features.features import _Feature
from ..features.types import MappingType, Type
from ..typing import Feature

DataFlow: TypeAlias = object


# TODO: legacy code
class IOContext:
    ...


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

    rank: Rank

    input_type: MappingType

    output_type: Type

    def __hash__(self) -> int:
        """Returns a hash value based on the node ID.

        Returns:
            int: The hash value of the node ID.
        """
        return hash(self.node_id)


Params = ParamSpec("Params")
Return = TypeVar("Return", covariant=True)


class NodeProtocol(Protocol, Generic[Params, Return]):
    def arguments(self) -> set[str]:
        ...

    @overload
    def call(self, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        ...

    @overload
    def call(self, flow: AbstractDataFlow, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        ...

    def call(self, *args: Params.args, **kwargs: Params.kwargs) -> Return:
        ...


class BaseNodeConfig(BaseConfig):
    """Base configuration class for nodes in a data flow graph."""


C = TypeVar("C", bound=BaseNodeConfig)


class BaseNode(BaseConfigurable[C], ABC):
    """Base class for nodes in a data flow graph."""

    DEFAULT_OUTPUT_KEY: str = "output"

    @classmethod
    @property
    def Config(self) -> type[C]:
        """Get the configuration type of the node."""
        return self.config_type

    @abstractmethod
    def signature(self) -> Signature:
        ...

    @overload
    def call(self, *args: Feature, **kwargs: Feature) -> _Feature:
        ...

    @overload
    def call(self, flow: AbstractDataFlow, *args: Feature, **kwargs: Feature) -> _Feature:
        ...

    def _extract_graph_from_args(
        self, args: tuple[AbstractDataFlow | Feature], kwargs: dict[str, AbstractDataFlow | Feature]
    ) -> tuple[AbstractDataFlowGraph, tuple[Feature], dict[str, Feature]]:
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
            raise RuntimeError("Flow cannot be inferred from arguments!")

        # get the flow from the reference
        return reference.ref._graph, args, kwargs

    def call(
        self,
        *args: AbstractDataFlow | Feature,
        **kwargs: AbstractDataFlow | Feature,
    ) -> _Feature:
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
