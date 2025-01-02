"""Type Validators and Resolvers.

This module defines custom validators and resolvers for validating and resolving data types
using Pydantic's validation framework. These classes extend Pydantic's :class:`AfterValidator`
and :class:`BeforeValidator` to enforce validation and type resolution logic.
"""


from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Callable, TypeVar
from uuid import UUID

from pydantic import AfterValidator, BeforeValidator, TypeAdapter, ValidationInfo

from ..registry.config import BaseConfig
from .dtypes import Type
from .features import Feature, SequenceFeature, build_feature_from_reference
from .reference import BaseReference


@dataclass(eq=True, frozen=True)
class TypeValidator(AfterValidator):
    """A Pydantic :class:`AfterValidator` class that performs type validation.

    Raises:
        RuntimeError: If the context does not provide the required :code:`'config'` and
        :code:`'session_id'`.
    """

    def __init__(self, validator: Callable[[Any, BaseConfig, UUID], Any]) -> None:
        """Initializes the :class:`TypeValidator` with a custom validation function.

        Args:
            validator (Callable[[Any, BaseConfig, UUID], Any]): The custom validation function
                to use.
        """

        def wrapped_validator(val: Any, info: ValidationInfo) -> Any:
            # context is required
            if (
                (info.context is None)
                or ("config" not in info.context)
                or ("session_id" not in info.context)
            ):
                raise RuntimeError()

            return validator(val, info.context["config"], info.context["session_id"])

        super(TypeValidator, self).__init__(wrapped_validator)


@dataclass(eq=True, frozen=True)
class TypeResolver(BeforeValidator):
    """A Pydantic :class:`BeforeValidator` that resolves ambiguous types.

    Raises:
        RuntimeError: If the context does not provide the required :code:`'config'`,
            :code:`'inputs'`, or :code:`'session_id'`.
    """

    def __init__(self, resolver: Callable[[BaseConfig, dict[str, Feature], UUID], Any]) -> None:
        """Initializes the :class:`TypeResolver` with a custom resolver function.

        Args:
            resolver (Callable[[BaseConfig, dict[str, Feature], UUID], Any]): The custom resolver
                function to use. The function receives the node configuration, the input features
                and the session id and returns the resolved feature type. The resolved feature type
                can be a feature class or annotation, a typevar or an instance of a :class:`Type`.
        """

        def wrapped_resolver(val: Any, info: ValidationInfo) -> Any:
            if isinstance(val, Feature):
                return val

            if isinstance(val, BaseReference) and val.get_dtype() is not None:
                return build_feature_from_reference(val)

            # context is required
            if (
                (info.context is None)
                or ("config" not in info.context)
                or ("inputs" not in info.context)
                or ("session_id" not in info.context)
                or ("typevars" not in info.context)
            ):
                raise RuntimeError("Missing context")

            # create an instance of the target type
            target_type = resolver(
                info.context["config"], info.context["inputs"], info.context["session_id"]
            )

            # resolve type variable
            if isinstance(target_type, TypeVar):
                typevars = {str(var): typ for var, typ in info.context["typevars"].items()}
                if str(target_type) not in typevars.keys():
                    raise RuntimeError(f"Invalid TypeVar {target_type}")

                target_type = typevars[str(target_type)]
                assert isinstance(target_type, Type)

            # create feature from dtype
            if isinstance(target_type, Type):
                target_type = Annotated[
                    Feature,
                    BeforeValidator(
                        partial(build_feature_from_reference, fallback_dtype=target_type)
                    ),
                ]

            return TypeAdapter(target_type).validate_python(val, context=info.context)

        super(TypeResolver, self).__init__(wrapped_resolver)


@dataclass(eq=True, frozen=True)
class Len(TypeValidator):
    """A :class:`TypeValidator` that checks the length of a sequence."""

    def __init__(self, length: int) -> None:
        """Initializes the :class:`Len` validator with the expected length of the sequence.

        Args:
            length (int): The expected length of the sequence to be validated.
        """

        def length_validator(seq: SequenceFeature, config: BaseConfig, val_id: UUID) -> Any:
            if not isinstance(seq, SequenceFeature):
                raise RuntimeError("Not a sequence")

            if length != seq.length():
                raise TypeError(f"Length mismatch, {length} != {seq.length()}")

            return seq

        super(Len, self).__init__(length_validator)
