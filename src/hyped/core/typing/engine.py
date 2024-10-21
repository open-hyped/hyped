"""Type Handling and Validation Module.

This module provides utilities for type validation, type variable registration, and type checking
within a data flow graph system. It allows dynamic handling of type annotations, argument
validation, and captures type variables during function calls. It integrates with Pydantic to
handle model validation.
"""
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from types import GenericAlias
from typing import Any, Callable, Generator, Generic, TypeVar, get_args, get_type_hints
from uuid import UUID, uuid4

import pydantic

from hyped._registry.config import BaseConfig
from hyped.common.feature_key import FeatureKey
from hyped.common.typing import DataFlowGraphAlias, NodeId

from .types import _Feature, _TypeFactory

_Type = TypeVar("_Type")


@dataclass
class _Factory(Generic[_Type]):
    """A factory class to instantiate and validate objects based on a type hint, using Pydantic."""

    hint: type[_Type]
    """The expected type of the instance."""

    config: BaseConfig
    """Configuration related to the data flow."""

    inputs: dict[str, _Feature]
    """Input data features used in validation."""

    session_id: UUID
    """Unique identifier for the validation session."""

    def __call__(self, _ptr: FeatureKey, _node_id: NodeId, _flow: DataFlowGraphAlias) -> _Type:
        """Create and validate an instance based on the provided arguments.

        Args:
            _ptr (FeatureKey): Key representing the feature pointer.
            _node_id (NodeId): The ID of the node in the data flow graph.
            _flow (DataFlowGraphAlias): Alias for the data flow graph.

        Returns:
            _Type: An instance of the validated type.
        """
        inst = {"_ptr": _ptr, "_node_id": _node_id, "_flow": _flow}
        context = {"config": self.config, "inputs": self.inputs, "session_id": self.session_id}
        return pydantic.TypeAdapter(self.hint).validate_python(inst, context=context)


class TypeVarRegister(object):
    """A registry for managing and capturing TypeVars used in type validation."""

    def __init__(self):
        """Initialize the TypeVarRegister."""
        self._registered_vars: dict[uuid4, TypeVar] = {}
        self._captured_vars: None | dict[TypeVar, Any] = None

    @contextmanager
    def capture(self) -> Generator[dict[TypeVar, Any], None, None]:
        """Context manager capturing TypeVar assignments during validation.

        Yields:
            dict[TypeVar, Any]: A dictionary mapping TypeVars to their assigned types.
        """
        self._captured_vars = {}
        yield self._captured_vars
        self._captured_vars = None

    def create_validator(self):
        """Create a Pydantic validator to validate and capture TypeVars.

        Returns:
            Callable: A Pydantic validator function with an associated UUID.
        """
        validator = partial(self._validator, uuid=uuid4())
        return pydantic.AfterValidator(validator)

    def register(self, T: TypeVar, validator: pydantic.AfterValidator) -> None:
        """Register a TypeVar with a Pydantic validator.

        Args:
            T (TypeVar): The TypeVar to register.
            validator (pydantic.AfterValidator): The validator associated with the TypeVar.
        """
        uuid = validator.func.keywords["uuid"]
        self._registered_vars[uuid] = T

    def _validator(self, val: object, uuid: uuid4) -> None:
        """Validate and capture a TypeVar during validation.

        Args:
            val (object): The value being validated.
            uuid (UUID): The UUID associated with the TypeVar.
        """

        # only active when values should be captured
        if self._captured_vars is None:
            return

        # get the typevar instance to validate
        T = self._registered_vars[uuid]
        # capture the value type
        if (T not in self._captured_vars) or (
            # prefer features over constants
            not isinstance(self._captured_vars[T], _Feature)
            and isinstance(val, _Feature)
        ):
            self._captured_vars[T] = val._type_hint if isinstance(val, _Feature) else type(val)

        # TODO: make sure the value matches the captured type

        return val


type_var_register = TypeVarRegister()
"""Global type variable register instance."""


class TypeEngine(object):
    def __init__(self, name: str, config: BaseConfig, func: Callable) -> None:
        """Initialize the TypeEngine with a function and a configuration.

        Args:
            name (str): A name used as an identifier for context in logs and errors.
            config (BaseConfig): A configuration object provided as context during validation.
            func (Callable): The callable to analyse.
        """

        self.name = name
        self.config = config
        # parse the function
        self.signature = inspect.signature(func)
        self.hints = get_type_hints(func, include_extras=True)
        # create the validation model
        self.session_id = uuid4()
        self.validator = self._build_validator(func)
        # track the type variable assingment while validating
        self.vars_mapping: None | dict[TypeVar, Any] = None

        self.args_param = next(
            filter(
                lambda p: p.kind is inspect._ParameterKind.VAR_POSITIONAL,
                self.signature.parameters.values(),
            ),
            None,
        )
        self.kwargs_param = next(
            filter(
                lambda p: p.kind is inspect._ParameterKind.VAR_KEYWORD,
                self.signature.parameters.values(),
            ),
            None,
        )

    def _build_validator(self, func: Callable) -> pydantic.BaseModel:
        """Build a Pydantic model to validate function arguments based on type hints.

        Args:
            func (Callable): The function whose arguments are being validated.

        Returns:
            pydantic.BaseModel: The argument validator model.
        """

        # get all input argument type hints
        hints = self.hints.copy()
        hints.pop("return")

        arguments = {}
        # build arguments from type annotations
        for name, annotation in hints.items():
            param = self.signature.parameters[name]
            annotation = (
                list[annotation]
                if param.kind is inspect._ParameterKind.VAR_POSITIONAL
                else dict[str, annotation]
                if param.kind is inspect._ParameterKind.VAR_KEYWORD
                else annotation
            )
            arguments[name] = (annotation, pydantic.Field())
        # build input argument validator model
        return pydantic.create_model(f"ArgumentValidator({func.__name__})", **arguments)

    def validate_signature(self) -> None:
        """Validate that the function signature has all necessary type annotations.

        Raises:
            TypeError: If any parameter is missing a type annotation.
        """

        # TODO
        if self.args_param is not None:
            raise NotImplementedError()

        for name, param in self.signature.parameters.items():
            if param.annotation is inspect._empty:
                raise TypeError(
                    f"Missing type annotation for parameter '{name}' in function '{self.name}'. "
                    "All parameters must have type annotations to ensure proper validation."
                )

    def validate_arguments(self, *args: Any, **kwargs: Any) -> dict[str, _Feature]:
        """Validate the arguments passed to the function based on its signature.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            dict[str, _Feature]: Validated input features.

        Raises:
            TypeError: If the arguments provided are invalid or do not match the expected types.
        """

        # bind arguments to signature
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        try:
            # validate input arguments
            with type_var_register.capture() as self.vars_mapping:
                context = {"config": self.config, "session_id": self.session_id}
                self.validator.model_validate(bound_args.arguments, context=context)

        except pydantic.ValidationError as e:
            # TODO: improve error message to include error keys and expected type
            raise TypeError(
                f"Invalid argument types provided in the call to '{self.name}'. "
            ) from e

    def get_inputs_and_consts(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, _Feature], dict[str, Any], dict[str, _TypeFactory]]:
        """
        Separate input features and constants from the arguments.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            tuple[dict[str, _Feature], dict[str, Any], dict[str, _TypeFactory]]: Tuple containing
            input features and constants.
        """

        arguments = self.signature.bind(*args, **kwargs).arguments
        kwargs = arguments.pop(self.kwargs_param.name) if self.kwargs_param is not None else {}
        arguments.update(kwargs)
        # separate all feature and constant inputs
        inputs = {key: val for key, val in arguments.items() if isinstance(val, _Feature)}
        consts = {key: val for key, val in arguments.items() if not isinstance(val, _Feature)}
        # get the type hints for the constants from the signature
        const_factories = {
            key: self.hints[key if key not in kwargs else self.kwargs_param.name]
            for key in consts.keys()
        }
        const_factories = {
            key: self._build_type_factory(hint, arguments) for key, hint in const_factories.items()
        }

        return inputs, consts, const_factories

    def get_return_factory(self, inputs: dict[str, _Feature]) -> _TypeFactory:
        """Build a type factory for the return type based on the inputs.

        Args:
            inputs (dict[str, _Feature]): Input features for return type validation.

        Returns:
            _TypeFactory: A factory class for the return type.
        """
        return self._build_type_factory(self.hints["return"], inputs)

    def _build_type_factory(self, hint: type, inputs: None | dict[str, Any] = None) -> _TypeFactory:
        """Build a factory for a type based on the given hint and input features.

        Args:
            hint (type): Type hint for the factory.
            inputs (Optional[dict[str, Any]]): Input features used in validation.

        Returns:
            _TypeFactory: Factory class for creating instances of the specified type.

        Raises:
            TypeError: If the type hint provided is invalid or not supported.
        """

        if isinstance(hint, GenericAlias):
            args = get_args(hint)
            args = list(self.vars_mapping[arg] for arg in args if isinstance(arg, TypeVar))
            hint = hint.__class_getitem__(*args) if len(args) > 0 else hint
            return _Factory(hint, config=self.config, inputs=inputs, session_id=self.session_id)

        if isinstance(hint, TypeVar):
            return self.build_type_factory(self.vars_mapping[hint], inputs=inputs)

        if issubclass(hint, _Feature):
            return _Factory(hint, config=self.config, inputs=inputs, session_id=self.session_id)

        raise TypeError(
            f"Invalid type hint '{hint}' provided in function '{self.name}'. "
            "Expected a GenericAlias, TypeVar, or subclass of _Feature, but received "
            f"'{type(hint).__name__}'. Please check the type hint and ensure it is supported."
        )
