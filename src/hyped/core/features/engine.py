"""Type Handling and Validation Module.

This module provides utilities for type validation, type variable registration, and type checking
within a data flow graph system. It allows dynamic handling of type annotations, argument
validation, and captures type variables during function calls. It integrates with Pydantic to
handle model validation.
"""
import inspect
from functools import partial
from typing import Annotated, Any, Generic, TypeVar
from uuid import UUID, uuid4

import pydantic
import pydantic.generics

from hyped._registry.config import BaseConfig

from .features import _Feature, build_feature_from_annotation
from .reference import Reference
from .types import Type


class TypeVarRegister(object):
    """A registry for managing and capturing TypeVars used in type validation."""

    def __init__(self):
        """Initialize the TypeVarRegister."""
        self._registered_vars: dict[uuid4, TypeVar] = {}
        self._captured_vars: dict[TypeVar, Type | type] = {}

    @property
    def typevar_mapping(self) -> dict[TypeVar, Type]:
        # TODO: map python build-in types to data types
        if any(not isinstance(dtype, Type) for dtype in self._captured_vars.values()):
            raise NotImplementedError()

        return self._captured_vars

    def solve_typevar(self, var: TypeVar) -> Type | type:
        return self.typevar_mapping[var]

    def create_trackable_typevar(self, *args, **kwargs) -> TypeVar:
        validator = self.create_validator()
        # annotate the bound argument with the validator
        bound = kwargs.pop("bound", Any)
        bound = Annotated[bound, validator]
        # create the typevar
        T = TypeVar(*args, bound=bound, **kwargs)
        # register the typevar
        self.register(T, validator)

        return T

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

    def _validator(self, val: object, uuid: UUID) -> None:
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
            self._captured_vars[T] = val.dtype if isinstance(val, _Feature) else type(val)

        # TODO: make sure the value matches the captured type

        return val


class FeatureEngine(object):
    def __init__(self, name: str, config: BaseConfig, signature: inspect.Signature) -> None:
        self.name = name
        self.config = config

        self.signature = signature
        # track the type variable assingment while validating
        self.typevar_register = TypeVarRegister()
        # create the validation model
        self.session_id = uuid4()
        self.validator, self.typevar_lookup = self._build_validator()

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

        params = set()
        # collect all typevars in all annotations
        for arg, _ in arguments.values():
            if isinstance(arg, TypeVar):
                params.add(arg)
            elif hasattr(arg, "__parameters__"):
                params.update(arg.__parameters__)

        # create fixed order over parameters
        params = tuple(params)

        base = (pydantic.BaseModel,)
        if len(params) > 0:
            base += (Generic[params],)

        # build input argument validator model
        validator = pydantic.create_model(
            f"ArgumentValidator({self.name})", **arguments, __base__=base
        )

        # create trackable typevars
        typevars = tuple(
            self.typevar_register.create_trackable_typevar(
                p.__name__, bound=p.__bound__ if p.__bound__ is not None else Any
            )
            for p in params
        )

        if len(params) > 0:
            # apply typevars to validator model
            validator = validator.__class_getitem__(*typevars)

        return validator, dict(zip(params, typevars))

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
            context = {"config": self.config, "session_id": self.session_id}
            self.validator.model_validate(bound_args.arguments, context=context)

        except pydantic.ValidationError as e:
            # TODO: improve error message to include error keys and expected type
            raise TypeError(
                f"Invalid argument types provided in the call to '{self.name}'. "
            ) from e

    def get_references_and_consts(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Reference], dict[str, Any], dict[str, Type]]:
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
        inputs = {key: val.ref for key, val in arguments.items() if isinstance(val, _Feature)}
        consts = {key: val for key, val in arguments.items() if not isinstance(val, _Feature)}
        # get the type hints for the constants from the signature
        {
            key: self.signature.parameters[
                key if key not in kwargs else self.kwargs_param.name
            ].annotation
            for key in consts.keys()
        }
        # TODO: infer data type from signature
        const_types = {}

        return inputs, consts, const_types

    def build_return_feature(self, ref: Reference, inputs: dict[str, _Feature]) -> _Feature:
        # get the return annotation
        annotation = self.signature.return_annotation

        typevar_mapping = {
            t: self.typevar_register.solve_typevar(u) for t, u in self.typevar_lookup.items()
        }

        context = {
            "inputs": inputs,
            "config": self.config,
            "session_id": self.session_id,
            "typevars": self.typevar_register.typevar_mapping,
        }

        return build_feature_from_annotation(ref, annotation, typevar_mapping, context)
