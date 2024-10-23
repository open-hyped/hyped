from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Generic, TypeVar, get_args
from uuid import UUID, uuid4

import pydantic

from hyped._registry.config import BaseConfig
from hyped.common.typing import ArrowType, NodeId
from hyped.core.features.feature_key import FeatureKey

from ..abstract import AbstractDataFlowGraph

T = TypeVar("T")


class _DummyDataFlowGraph(AbstractDataFlowGraph):
    def __init__(self):
        pass


class BaseFeatureFactory(ABC, Generic[T]):
    @property
    def _pa_type(self) -> ArrowType:
        return self.instance._pa_type

    @property
    def instance(self) -> T:
        graph = _DummyDataFlowGraph()
        return self(FeatureKey(), "NodeId", graph)

    @abstractmethod
    def __call__(self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph) -> T:
        """Create and validate an instance based on the provided arguments.

        Args:
            _key (FeatureKey): Key representing the feature pointer.
            _node_id (NodeId): The ID of the node in the data flow graph.
            _graph (AbstractDataFlowGraph): The data flow graph containing the Feature.

        Returns:
            T: The type instance generated from the arguments.
        """
        ...


@dataclass
class DefaultFeatureFactory(BaseFeatureFactory[T]):
    config: BaseConfig = field(default_factory=BaseConfig)
    """Configuration related to the data flow."""

    inputs: dict[str, Any] = field(default_factory=dict)
    """Input data features used in validation."""

    session_id: UUID = field(default_factory=uuid4)
    """Unique identifier for the validation session."""

    typevars: dict[TypeVar, type] = field(default_factory=dict)
    """Typevar lookup used in type resolvers when typevars occur."""

    def __call__(self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph) -> T:
        (hint,) = get_args(self.__orig_class__)
        assert not isinstance(hint, TypeVar)

        inst = {"_key": _key, "_node_id": _node_id, "_graph": _graph}
        context = {
            "config": self.config,
            "inputs": self.inputs,
            "session_id": self.session_id,
            "typevars": self.typevars,
        }
        return pydantic.TypeAdapter(hint).validate_python(inst, context=context)


@dataclass
class FeatureFactory(BaseFeatureFactory[T]):
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph) -> T:
        (hint,) = get_args(self.__orig_class__)
        assert not isinstance(hint, TypeVar) and isinstance(hint, type)

        inst = {"_key": _key, "_node_id": _node_id, "_graph": _graph}
        return hint(**(self.kwargs | inst))


@dataclass
class FeatureFactoryFromInstance(FeatureFactory[T]):
    def __init__(self, inst: T) -> None:
        kwargs = {field.name: getattr(self.inst, field.name) for field in fields(self.inst)}
        kwargs.pop("_key")
        kwargs.pop("_node_id")
        kwargs.pop("_graph")

        super(FeatureFactoryFromInstance, self).__init__(**kwargs)
