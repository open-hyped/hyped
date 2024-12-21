"""Type Handling and Validation Module.

This module provides utilities for type validation, type variable registration, and type checking
within a data flow graph system. It allows dynamic handling of type annotations, argument
validation, and captures type variables during function calls. It integrates with Pydantic to
handle model validation.
"""
import inspect
from collections import defaultdict
from functools import partial
from typing import Annotated, Any, Generic, TypeVar
from uuid import UUID, uuid4

import pydantic
import pydantic.generics

from hyped.common._pydantic import BaseModelWithArbitraryTypesAllowed

from ..registry.config import BaseConfig
from .dtypes import Type, build_dtype_from_python_object, common_dtype
from .features import Feature, build_feature_from_annotation
from .reference import Reference


class TypeVarRegister(object):
    """A registry for managing and capturing TypeVars used in type validation."""

    def __init__(self):
        """Initialize the TypeVarRegister."""
        self._registered_vars: dict[UUID, TypeVar] = {}
        self._captured_vars: dict[TypeVar, set[Type]] = defaultdict(set)

    def reset(self) -> None:
        """Reset the type var register."""
        self._captured_vars.clear()

    @property
    def typevar_mapping(self) -> dict[TypeVar, Type]:
        """Map captured TypeVars to their resolved types.

        Returns:
            dict[TypeVar, type]: A dictionary mapping each registered TypeVar
            to its resolved type.
        """
        return {var: self.solve_typevar(var) for var in self._captured_vars.keys()}

    def solve_typevar(self, var: TypeVar) -> Type:
        """Resolve a TypeVar to its captured type.

        Args:
            var (TypeVar): The :class:`TypeVar` to resolve.

        Returns:
            type: The captured type associated with the provided :class:`TypeVar`.

        Raises:
            KeyError: If the TypeVar is not found in the captured variables.
        """
        return common_dtype(*self._captured_vars[var])

    def create_trackable_typevar(self, *args: Any, **kwargs: Any) -> TypeVar:
        """Create and register a TypeVar with a custom validator.

        Args:
            *args (Any): Positional arguments passed to the :class:`TypeVar` constructor.
            **kwargs (Any): Keyword arguments passed to the :class:`TypeVar` constructor.
                Special :code:`bound` keyword is used to define the type bound for the TypeVar.

        Returns:
            TypeVar: The newly created and registered :class:`TypeVar` with the custom validator.
        """
        validator = self.create_validator()
        # annotate the bound argument with the validator
        bound = kwargs.pop("bound", Any)
        bound = Annotated[bound, validator]
        # create the typevar
        T = TypeVar(*args, bound=bound, **kwargs)  # type: ignore
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

    def register(self, t: TypeVar, validator: pydantic.AfterValidator) -> None:
        """Register a TypeVar with a Pydantic validator.

        Args:
            t (TypeVar): The TypeVar to register.
            validator (pydantic.AfterValidator): The validator associated with the TypeVar.
        """
        uuid = validator.func.keywords["uuid"]
        self._registered_vars[uuid] = t

    def _validator(self, val: object, uuid: UUID) -> None:
        """Capture a TypeVar during validation.

        Args:
            val (object): The value being validated.
            uuid (UUID): The UUID associated with the TypeVar.
        """
        # get the typevar instance to validate
        t = self._registered_vars[uuid]
        # capture the value type
        if (t not in self._captured_vars) or (
            # prefer features over constants
            not isinstance(self._captured_vars[t], Type)
            and isinstance(val, Feature)
        ):
            dtype = val.dtype if isinstance(val, Feature) else build_dtype_from_python_object(val)
            self._captured_vars[t].add(dtype)

        return val


class FeatureEngine(object):
    """Feature Engine.

    The :class:`FeatureEngine` ensures that the input features conform to the expected types
    defined in the function's signature and uses the type annotations to construct output
    features dynamically.
    """

    def __init__(self, name: str, config: BaseConfig, signature: inspect.Signature) -> None:
        """Initialize the :class:`FeatureEngine`.

        Args:
            name (str): The name of the feature engine or associated function.
            config (BaseConfig): Configuration object for feature creation and validation.
            signature (inspect.Signature): Function signature to validate arguments against.
        """
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

        base = (BaseModelWithArbitraryTypesAllowed,)
        if len(params) > 0:
            base += (Generic[params],)  # type: ignore

        # build input argument validator model
        validator = pydantic.create_model(
            f"ArgumentValidator({self.name})",
            **arguments,
            __base__=base,
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

        return validator, dict(zip(params, typevars, strict=True))

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

        if self.signature.return_annotation is inspect._empty:
            raise TypeError("Mussing return type annotation in function '{self.name}'.")

    def validate_arguments(self, *args: Any, **kwargs: Any) -> None:
        """Validate the arguments passed to the function based on its signature.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Raises:
            TypeError: If the arguments provided are invalid or do not match the expected types.
        """
        self.typevar_register.reset()
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

    def get_references_and_objects(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Reference], dict[str, Any], dict[str, Type]]:
        """Separate input feature references and constants from the arguments.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            tuple[dict[str, _Feature], dict[str, Any], dict[str, TypeFactory]]: Tuple containing
            input features and objects.
        """
        # bind inputs to signature and extract the keyword arguments
        arguments = self.signature.bind(*args, **kwargs).arguments
        kwargs = arguments.pop(self.kwargs_param.name, {}) if self.kwargs_param is not None else {}
        # include the keyword arguments in the arguments
        arguments.update(kwargs)

        # separate all feature and constant inputs
        inputs = {key: val.ref for key, val in arguments.items() if isinstance(val, Feature)}
        consts = {key: val for key, val in arguments.items() if not isinstance(val, Feature)}

        # infer the data types of the constant inputs from the signature
        const_dtypes = {
            key: self.build_feature_with_context(
                annotation=self.signature.parameters[
                    key if key not in kwargs else self.kwargs_param.name
                ].annotation,
                ref=Reference(),
                inputs=None,
            ).dtype
            for key in consts.keys()
        }

        return inputs, consts, const_dtypes

    def build_feature_with_context(
        self, annotation: Any, ref: Reference, inputs: None | dict[str, Feature]
    ) -> Feature:
        """Create a feature based on type annotation and inputs in a specific context.

        Args:
            annotation (Any): Type annotation for the feature.
            ref (Reference): Reference to the feature being created.
            inputs (None | dict[str, Feature]): Input features to build the feature, if any.

        Returns:
            Feature: The constructed feature.
        """
        typevar_mapping = {
            t: self.typevar_register.solve_typevar(u) for t, u in self.typevar_lookup.items()
        }

        context = {
            "config": self.config,
            "session_id": self.session_id,
            "typevars": self.typevar_register.typevar_mapping,
        }

        if inputs is not None:
            context["inputs"] = inputs

        return build_feature_from_annotation(ref, annotation, typevar_mapping, context)

    def build_return_feature(self, ref: Reference, inputs: dict[str, Feature]) -> Feature:
        """Build the return feature based on the function's return type annotation.

        Args:
            ref (Reference): Reference to the return feature.
            inputs (dict[str, Feature]): Input features for constructing the return feature.

        Returns:
            Feature: The constructed return feature.
        """
        return self.build_feature_with_context(self.signature.return_annotation, ref, inputs)
