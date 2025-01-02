"""Type Validators and Resolvers.

This module defines custom validators and resolvers for validating and resolving data types
using Pydantic's validation framework. These classes extend Pydantic's :class:`AfterValidator`
and :class:`BeforeValidator` to enforce validation and type resolution logic.
"""


from dataclasses import dataclass, replace
from functools import partial
from typing import Annotated, Any, Callable, TypeAlias
from uuid import UUID

from pydantic import AfterValidator, BeforeValidator, ValidationInfo

from ..registry.config import BaseConfig
from .dtypes import UNDEFINED_SEQUENCE_LENGTH, Type
from .features import (
    Feature,
    SequenceFeature,
    build_feature_from_annotation,
    build_feature_from_reference,
)
from .reference import BaseReference, ForwardReference
from .session import ValidationSession

ValidatorFunc: TypeAlias = Callable[[Any, None | BaseConfig, ValidationSession], Any]
ResolverFunction: TypeAlias = Callable[[BaseConfig, dict[str, Feature], UUID], Any]


@dataclass(eq=True, frozen=True)
class FeatureValidator(AfterValidator):
    """A Pydantic :class:`AfterValidator` class that performs feature validation.

    Raises:
        RuntimeError: If the context does not provide the required :code:`'config'` and
        :code:`'session'`.
    """

    def __init__(self, validator: ValidatorFunc) -> None:
        """Initializes the :class:`TypeValidator` with a custom validation function.

        Args:
            validator (ValidatorFunc): The custom validation function to use. Takes the
                feature to validate, the config of the corresponding node and the validation
                session instance as input and returns the validated feature instance.
        """

        def wrapped_validator(val: Any, info: ValidationInfo) -> Any:
            # context is required
            if (
                (info.context is None)
                or ("config" not in info.context)
                or ("session" not in info.context)
            ):
                raise RuntimeError()

            return validator(val, info.context["config"], info.context["session"])

        super(FeatureValidator, self).__init__(wrapped_validator)


@dataclass(eq=True, frozen=True)
class FeatureResolver(BeforeValidator):
    """A Pydantic :class:`BeforeValidator` that resolves ambiguous feature types.

    Raises:
        RuntimeError: If the context does not provide the required :code:`'config'`,
            :code:`'inputs'`, or :code:`'session'`.
    """

    def __init__(self, resolver: ResolverFunction) -> None:
        """Initializes the :class:`TypeResolver` with a custom resolver function.

        Args:
            resolver (ResolverFunction): The custom resolver function to use. The function
                receives the node configuration, the input features and the session id and
                returns the resolved feature type. The resolved feature type can be a feature
                class or annotation, a typevar or an instance of a :class:`Type`.
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
                or ("session" not in info.context)
                or ("typevars" not in info.context)
            ):
                raise RuntimeError("Missing context")

            # create an instance of the target type
            target_type = resolver(
                info.context["config"], info.context["inputs"], info.context["session"]
            )

            # create feature from dtype
            if isinstance(target_type, Type):
                target_type = Annotated[
                    Feature,
                    BeforeValidator(
                        partial(build_feature_from_reference, fallback_dtype=target_type)
                    ),
                ]

            return build_feature_from_annotation(
                target_type, info.context["typevars"], info.context["session"], info.context
            )

        super(FeatureResolver, self).__init__(wrapped_resolver)


@dataclass(eq=True, frozen=True)
class Len(FeatureValidator):
    """A :class:`TypeValidator` that checks the length of a sequence."""

    def __init__(self, length: None | int = None) -> None:
        """Initializes the :class:`Len` validator with the expected length of the sequence.

        Args:
            length (int | None): The expected length of the sequence to be validated.
        """

        def length_validator(
            seq: SequenceFeature, config: BaseConfig, session: ValidationSession
        ) -> Any:
            if not isinstance(seq, SequenceFeature):
                raise RuntimeError("Not a sequence")

            # get the expected length, potentially from the session context
            actual_length = len(seq.dtype)
            expected_length = length or session.get_context(self)

            if (actual_length != UNDEFINED_SEQUENCE_LENGTH) and (expected_length is not None):
                # sequence feature length and expected length are both well defined
                # so check if they match up
                if expected_length != actual_length:
                    raise TypeError(f"Length mismatch, {length} != {actual_length}")

            elif (actual_length != UNDEFINED_SEQUENCE_LENGTH) and (expected_length is None):
                # sequence feature length is well defined but expected length is not
                # so capture the length of the sequence for later usage
                session.set_context(self, actual_length)

            elif (
                (actual_length == UNDEFINED_SEQUENCE_LENGTH)
                and (expected_length is not None)
                and isinstance(seq.ref, ForwardReference)
            ):
                # specifies the length of the sequence
                dtype = replace(seq.dtype, length=expected_length)
                return SequenceFeature(ForwardReference(dtype))

            return seq

        super(Len, self).__init__(length_validator)
