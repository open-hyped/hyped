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

from hyped._registry.config import BaseConfig

from .features import Sequence, _Feature, build_feature_from_dtype
from .types import Type


@dataclass(eq=True, frozen=True)
class TypeValidator(AfterValidator):
    """A Pydantic :class:`AfterValidator` class that performs type validation.

    Raises:
        RuntimeError: If the context does not provide the required :code:`'config'` and
        :code:`'session_id'`.
    """

    def __init__(self, validator: Callable[[Any, BaseConfig], Any]) -> None:
        """
        Initializes the :class:`TypeValidator` with a custom validation function.

        Args:
            validator (Callable[[Any, BaseConfig], Any]): The custom validation function to use.
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

    def __init__(self, resolver: Callable[[BaseConfig, dict[str, _Feature]], Any]) -> None:
        """Initializes the :class:`TypeResolver` with a custom resolver function.

        Args:
            resolver (Callable[[BaseConfig, dict[str, _Feature]], Any]): The custom resolver
                function to use.
        """

        def wrapped_resolver(val: Any, info: ValidationInfo) -> Any:
            # type resolvers only apply when creating a type instance
            if isinstance(val, _Feature):
                return val

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
                if target_type not in info.context["typevars"].keys():
                    raise RuntimeError(f"Invalid TypeVar {target_type}")

                target_dtype = info.context["typevars"][target_type]
                assert isinstance(target_dtype, Type)

                target_type = Annotated[
                    _Feature, BeforeValidator(partial(build_feature_from_dtype, dtype=target_dtype))
                ]

            return TypeAdapter(target_type).validate_python(val, context=info.context)

        super(TypeResolver, self).__init__(wrapped_resolver)


@dataclass(eq=True, frozen=True)
class Len(TypeValidator):
    """A :class:`TypeValidator` that checks the length of a sequence."""

    def __init__(self, length: int) -> None:
        """
        Initializes the :class:`Len` validator with the expected length of the sequence.

        Args:
            length (int): The expected length of the sequence to be validated.
        """

        def length_validator(seq: Sequence, config: BaseConfig, val_id: UUID) -> Any:
            if not isinstance(seq, Sequence):
                raise RuntimeError("Not a sequence")

            if length != len(seq):
                raise TypeError(f"Length mismatch, {length} != {len(seq)}")

            return seq

        super(Len, self).__init__(length_validator)
