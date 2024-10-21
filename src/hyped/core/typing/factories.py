from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Generic, TypeVar, get_args
from uuid import UUID, uuid4

import pydantic

from hyped._registry.config import BaseConfig
from hyped.common.feature_key import FeatureKey
from hyped.common.typing import NodeId

from ..abstract import AbstractDataFlowGraph

T = TypeVar("T")


class BaseTypeFactory(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, _ptr: FeatureKey, _node_id: NodeId, _flow: AbstractDataFlowGraph) -> T:
        """Create and validate an instance based on the provided arguments.

        Args:
            _ptr (FeatureKey): Key representing the feature pointer.
            _node_id (NodeId): The ID of the node in the data flow graph.
            _flow (DataFlowGraphAlias): Alias for the data flow graph.

        Returns:
            T: The type instance generated from the arguments.
        """
        ...


@dataclass
class DefaultTypeFactory(BaseTypeFactory[T]):
    config: BaseConfig = field(default_factory=BaseConfig)
    """Configuration related to the data flow."""

    inputs: dict[str, Any] = field(default_factory=dict)
    """Input data features used in validation."""

    session_id: UUID = field(default_factory=uuid4)
    """Unique identifier for the validation session."""

    def __call__(self, _ptr: FeatureKey, _node_id: NodeId, _flow: AbstractDataFlowGraph) -> T:
        (hint,) = get_args(self.__orig_class__)
        assert not isinstance(hint, TypeVar)

        inst = {"_ptr": _ptr, "_node_id": _node_id, "_flow": _flow}
        context = {"config": self.config, "inputs": self.inputs, "session_id": self.session_id}
        return pydantic.TypeAdapter(hint).validate_python(inst, context=context)


@dataclass
class TypeFactoryFromInstance(BaseTypeFactory[T]):
    inst: T

    def __call__(self, _ptr: FeatureKey, _node_id: NodeId, _flow: AbstractDataFlowGraph) -> T:
        # get all fields from the dataclass instance
        kwargs = {field.name: getattr(self.inst, field.name) for field in fields(self.inst)}
        # overwrite the arguments
        kwargs["_ptr"] = _ptr
        kwargs["_node_id"] = _node_id
        kwargs["_flow"] = _flow
        # create a new instance using the keyword arguments
        return self.inst._type_hint(**kwargs)
