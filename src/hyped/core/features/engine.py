"""Type Handling and Validation Module.

This module provides utilities for type validation, type variable registration, and type checking
within a data flow graph system. It allows dynamic handling of type annotations, argument
validation, and captures type variables during function calls. It integrates with Pydantic to
handle model validation.
"""
import inspect
from contextlib import contextmanager
from functools import partial
from typing import Any, Generator, TypeVar
from uuid import uuid4

import pydantic

from hyped._registry.config import BaseConfig

from .factories import FeatureFactory
from .features import _Feature
from .reference import Reference


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
    def __init__(self, name: str, config: BaseConfig, signature: inspect.Signature) -> None:
        """Initialize the TypeEngine with a function and a configuration.

        Args:
            name (str): A name used as an identifier for context in logs and errors.
            config (BaseConfig): A configuration object provided as context during validation.
            signature (inspect.Signature): The function signature to consider for type checking.
        """

        self.name = name
        self.config = config

        self.signature = signature
        # create the validation model
        self.session_id = uuid4()
        self.validator = self._build_validator()
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

    def _build_validator(self) -> pydantic.BaseModel:
        """Build a Pydantic model to validate function arguments based on type hints.

        Returns:
            pydantic.BaseModel: The argument validator model.
        """

        arguments = {}
        # build arguments from type annotations
        for name, param in self.signature.parameters.items():
            annotation = (
                list[param.annotation]
                if param.kind is inspect._ParameterKind.VAR_POSITIONAL
                else dict[str, param.annotation]
                if param.kind is inspect._ParameterKind.VAR_KEYWORD
                else param.annotation
            )
            arguments[name] = (annotation, pydantic.Field())

        # build input argument validator model
        return pydantic.create_model(f"ArgumentValidator({self.name})", **arguments)

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

    def validate_arguments(self, *args: Any, **kwargs: Any) -> None:
        """Validate the arguments passed to the function based on its signature.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

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

    def get_references_and_consts(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, _Feature], dict[str, Any], dict[str, FeatureFactory]]:
        """
        Separate input feature references and constants from the arguments.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            tuple[dict[str, _Feature], dict[str, Any], dict[str, TypeFactory]]: Tuple containing
            input features and constants.
        """

        arguments = self.signature.bind(*args, **kwargs).arguments
        kwargs = arguments.pop(self.kwargs_param.name) if self.kwargs_param is not None else {}
        arguments.update(kwargs)
        # separate all feature and constant inputs
        inputs = {key: val for key, val in arguments.items() if isinstance(val, Reference)}
        consts = {key: val for key, val in arguments.items() if not isinstance(val, Reference)}
        # get the type hints for the constants from the signature
        const_annotations = {
            key: self.signature.parameters[
                key if key not in kwargs else self.kwargs_param.name
            ].annotation
            for key in consts.keys()
        }
        const_factories = {
            key: FeatureFactory[hint](
                config=self.config,
                inputs=arguments,
                session_id=self.session_id,
                typevars=self.vars_mapping,
            )
            for key, hint in const_annotations.items()
        }

        return inputs, consts, const_factories

    def get_return_factory(self, inputs: dict[str, _Feature]) -> FeatureFactory:
        """Build a type factory for the return type based on the inputs.

        Args:
            inputs (dict[str, _Feature]): Input features for return type validation.

        Returns:
            DefaultTypeFactory: A factory class for the return type.
        """
        return FeatureFactory[self.signature.return_annotation](
            config=self.config,
            inputs=inputs,
            session_id=self.session_id,
            typevars=self.vars_mapping,
        )
