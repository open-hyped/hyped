from dataclasses import dataclass, field, fields
from typing import Any, Generic, TypeVar, get_args
from uuid import UUID, uuid4

import pydantic

from hyped._registry.config import BaseConfig
from hyped.common.typing import ArrowType, NodeId

from ..abstract import AbstractDataFlowGraph
from .reference import FeatureKey

T = TypeVar("T")


class _DummyDataFlowGraph(AbstractDataFlowGraph):
    def __init__(self):
        pass


@dataclass
class FeatureFactory(Generic[T]):
    # TODO (docstring): feature factory with keyword arguments passed to
    # constructor, for ambiguous feature types the annotated type resolver
    # is used to disambiguate the feature type. In such cases the config,
    # inputs, session_id and typevar mapping might be relevant. For fully defined
    # types that do not require any type resolving or only the default type resolver
    # these arguments are not required.

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the feature constructor"""

    config: BaseConfig = field(default_factory=BaseConfig)
    """Configuration related to the data flow."""

    inputs: dict[str, Any] = field(default_factory=dict)
    """Input data features used in validation."""

    session_id: UUID = field(default_factory=uuid4)
    """Unique identifier for the validation session."""

    typevars: dict[TypeVar, type] = field(default_factory=dict)
    """Typevar lookup used in type resolvers when typevars occur."""

    strict: bool = True

    fallback_type_annotation: None | type[T] = None
    # type of the feature, by default inferred from the generic

    @property
    def _pa_type(self) -> ArrowType:
        return self.instance._pa_type

    @property
    def instance(self) -> T:
        graph = _DummyDataFlowGraph()
        return self(FeatureKey(), "NodeId", graph)

    def _build_validator_model(self, type_annotation: type) -> pydantic.BaseModel:
        base = (pydantic.BaseModel,)

        if isinstance(type_annotation, TypeVar):
            base += (Generic[type_annotation],)

        elif hasattr(type_annotation, "__parameters__") and (
            len(type_annotation.__parameters__) > 0
        ):
            base += (Generic[type_annotation.__parameters__],)

        # build pydantic model from type annotation for validation
        validator = pydantic.create_model(
            f"Validator({type_annotation})",
            field=(type_annotation, pydantic.Field()),
            __base__=base,
        )

        while hasattr(validator, "__parameters__") and len(validator.__parameters__) > 0:
            resolved_params = (self.typevars[t] for t in validator.__parameters__)
            validator = validator.__class_getitem__(*resolved_params)

        return validator

    def __call__(self, _key: FeatureKey, _node_id: NodeId, _graph: AbstractDataFlowGraph) -> T:
        """Create and validate an instance based on the provided arguments.

        Args:
            _key (FeatureKey): Key representing the feature pointer.
            _node_id (NodeId): The ID of the node in the data flow graph.
            _graph (AbstractDataFlowGraph): The data flow graph containing the Feature.

        Returns:
            T: The type instance generated from the arguments.
        """

        # get target feature type from annotation or fallback
        type_annotation = (
            get_args(self.__orig_class__)[0]
            if hasattr(self, "__orig_class__")
            else self.fallback_type_annotation
        )

        if type_annotation is None:
            # TODO (message): could not infer target type, at least one required
            raise RuntimeError()

        inst = self.kwargs | {"_key": _key, "_node_id": _node_id, "_graph": _graph}
        context = {
            "config": self.config,
            "inputs": self.inputs,
            "session_id": self.session_id,
            "typevars": self.typevars,
            "strict": self.strict,
        }

        try:
            validator = self._build_validator_model(type_annotation)
            return validator.model_validate({"field": inst}, context=context).field
        except pydantic.ValidationError as e:
            # TODO: error message
            raise RuntimeError() from e


class FeatureFactoryFromInstance(FeatureFactory[T]):
    def __init__(self, inst: T, typevars: dict[TypeVar, type] = {}, strict: bool = True) -> None:
        # get all the fields from the feature instance
        kwargs = {field.name: getattr(inst, field.name) for field in fields(inst)}
        # remove the key, node id and graph fields
        kwargs.pop("_key")
        kwargs.pop("_node_id")
        kwargs.pop("_graph")
        # create the feature factory instance
        super(FeatureFactoryFromInstance, self).__init__(
            kwargs=kwargs,
            typevars=typevars,
            strict=strict,
            fallback_type_annotation=inst._type_hint,
        )
