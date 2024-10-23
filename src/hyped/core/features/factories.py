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
        graph = _DummyDataFlowGraph()
        return self(FeatureKey(), "NodeId", graph)._pa_type

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
class FeatureFactoryFromInstance(BaseFeatureFactory[T]):
    inst: T

    def __post_init__(self) -> None:
        # overwrite reference values in instance
        self.inst = self.__call__(FeatureKey(), "NodeId", _DummyDataFlowGraph())

    @property
    def _pa_type(self) -> ArrowType:
        return self.inst._pa_type

    def __call__(self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph) -> T:
        # get all fields from the dataclass instance
        kwargs = {field.name: getattr(self.inst, field.name) for field in fields(self.inst)}
        # overwrite the arguments
        kwargs["_key"] = _key
        kwargs["_node_id"] = _node_id
        kwargs["_graph"] = _graph
        # create a new instance using the keyword arguments
        return self.inst._type_hint(**kwargs)
