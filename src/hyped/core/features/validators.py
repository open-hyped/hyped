"""Type Validators and Resolvers.

This module defines custom validators and resolvers for validating and resolving data types
using Pydantic's validation framework. These classes extend Pydantic's :class:`AfterValidator`
and :class:`BeforeValidator` to enforce validation and type resolution logic.
"""


from dataclasses import dataclass, replace
from functools import partial
from typing import Annotated, Any, Callable, Literal, TypeAlias, overload

from pydantic import AfterValidator, BeforeValidator, ValidationInfo
from pydantic_core import PydanticCustomError

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
ResolverFunction: TypeAlias = Callable[[BaseConfig, dict[str, Feature], ValidationSession], Any]


@dataclass(eq=True, frozen=True)
class FeatureValidator(AfterValidator):
    """A Pydantic :class:`AfterValidator` class that performs feature validation.

    This class is used to validate features in data processing pipelines. It integrates with
    Pydantic's :class:`AfterValidator` system to perform custom validation logic after the
    feature is processed. It is designed to be used with data flow systems where feature
    validation is required before performing any further operations.

    **Usage Example**:

    Below is an example of how to use :class:`FeatureValidator` to introduce custom feature
    validation logic into a data flow. The validator allows you to specify a custom validation
    function, which can be used with the :class:`Annotated` type hint.

    Example 1: **Basic Validation without Config**

    .. code-block:: python

        def custom_feature_validation(
            feature: Int,
            config: None,
            session: ValidationSession
        ) -> Int:
            # implement custom validation logic here
            return feature

        class Inputs(Mapping):
            a: Annotated[Int, FeatureValidator(custom_feature_validation)]
            b: Float

        flow = DataFlow[Inputs]()

    In this example, the feature :code:`a` is validated using the :class:`FeatureValidator` with
    the custom validation function :code:`custom_feature_validation`. The :code:`config` is
    :code:`None` because the validation are not specific to any node.

    Example 2: **Validation with Config in Processor Node**

    .. code-block:: python

        class CustomConfig(BaseDataProcessorConfig):
            value: int

        def custom_feature_validation(
            feature: Int,
            config: None | CustomConfig,
            session: ValidationSession
        ) -> Int:
            # implement custom validation logic here
            return feature

        class CustomProcessor(BaseDataProcessor[CustomConfig]):
            def process(
                self,
                ctx: RunContext,
                a: Annotated[Int, FeatureValidator(custom_feature_validation)]
            ) -> Float:
                # custom processing logic
                ...

    In this example, the feature :code:`a` is validated using the :class:`FeatureValidator`, but
    the validation function now has access to a :class:`CustomConfig` for the processor node,
    allowing for validation specific to the processor's configuration.
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
                raise RuntimeError(
                    "FeatureValidator requires 'config' and 'session' to be present "
                    "in the validation context. Ensure that these are provided."
                )

            return validator(val, info.context["config"], info.context["session"])

        super(FeatureValidator, self).__init__(wrapped_validator)


@dataclass(eq=True, frozen=True)
class FeatureResolver(BeforeValidator):
    """A Pydantic :class:`BeforeValidator` that resolves ambiguous feature types.

    This class is designed to resolve union types for features by allowing custom logic to
    determine the correct feature type before processing begins. It integrates with Pydantic's
    :class:`BeforeValidator` system to resolve any ambiguity in the feature's type. It is
    particularly useful when the feature type needs to change based on configuration or other
    contextual data.

    **Usage Example**:

    Below is an example of how to use :class:`FeatureResolver` to resolve ambiguous feature
    types based on a configuration or inputs. The validator allows you to specify a custom
    feature resolution function, which can be used with the :class:`Annotated` type hint.

    Example: **Resolving Union Types Based on Configuration**

    .. code-block:: python

        class CustomConfig(BaseDataProcessorConfig):
            value: int

        def custom_feature_resolution(
            config: None | CustomConfig,
            inputs: dict[str, Feature],
            session: ValidationSession
        ) -> Int:
            # implement custom resolution logic here
            return Int if config.value < 4 else Float

        class CustomProcessor(BaseDataProcessor[CustomConfig]):

            def process(
                self, ctx: RunContext, a: Int
            ) -> Annotated[Int | Float, FeatureResolver(custom_feature_resolution)]):
                # processing logic
                ...

    In this example, the :class:`FeatureResolver` is used to resolve the return feature type,
    which can be either :code:`Int` or :code:`Float`. The custom feature resolution function
    :code:`custom_feature_resolution` chooses the correct type based on the configuration.
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
                raise RuntimeError(
                    "FeatureResolver requires 'config', 'inputs', 'session', and 'typevars' "
                    "to be present in the validation context. Ensure that these are properly set."
                )

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


CapturedLength: TypeAlias = int | None
LengthComputeFn: TypeAlias = Callable[[BaseConfig, CapturedLength, ValidationSession], int | None]


@dataclass(eq=True, frozen=True)
class Len(FeatureValidator):
    """A :class:`TypeValidator` that sets/checks the length of a sequence.

    The :class:`Len` class is used to validate or specify the length of a sequence
    feature. Its behavior changes depending on whether the sequence's length is
    explicitly defined, whether the :code:`strict` mode is enabled, or whether
    it is used to ensure multiple sequences have the same length.

    **Usage Examples**:

    Example 1: **Specifying the Length of a Sequence**

    When the length of the sequence is not predefined, you can use :class:`Len` to specify it.
    This example shows how the :class:`Len` instance is used to enforce the sequence length.

    .. code-block:: python

        class Inputs(Mapping):
            seq: Annotated[Sequence[Int], Len(2)]

        flow = DataFlow[Inputs]()

    In this case, the sequence length of :code:`seq` will be set to 2.

    Example 2: **Conflict with Predefined Sequence Length**

    If the sequence length is already defined, the :class:`Len` instance will throw an exception
    when its length conflicts with the predefined length. Here's an example with a dataset where
    the length of :code:`seq` is already set to 1.

    .. code-block:: python

        features = datasets.Features()
        features["seq"] = datasets.Sequence(datasets.Value("Int32"), length=1)
        ds = datasets.Dataset.from_dict({"seq": [[0], [1]]}, features=features)

        class Inputs(Mapping):
            seq: Annotated[Sequence[Int], Len(2)]

        flow = DataFlow[Inputs](ds.features)

    In this example, the length of :code:`seq` is already specified as 1 in the dataset. When the
    :class:`Len` instance with length 2 is applied, an exception will be raised because the length
    2 does not match the predefined length 1.

    Example 3: **Ensuring Sequences Have the Same Length**

    You can use the :class:`Len` instance to ensure that multiple sequences have the same length
    without explicitly specifying the length. Here, :code:`match_length` is used to ensure that
    both :code:`seqA` and :code:`seqB` have the same length.

    .. code-block:: python

        match_length = Len()

        features = datasets.Features()
        features["seqA"] = datasets.Sequence(datasets.Value("Int32"), length=1)
        features["seqB"] = datasets.Sequence(datasets.Value("Int32"), length=1)

        class Inputs(Mapping):
            seqA: Annotated[Sequence[Int], match_length]
            seqB: Annotated[Sequence[Float], match_length]

        flow = DataFlow[Inputs](features)

    In this example, :code:`seqA` and :code:`seqB` are validated to ensure they both have the same
    length. If their lengths would differ, an exception will be raised.

    Example 4: **Dynamically Specifying the Length of a Sequence in a Processor Node**

    The :class:`Len` class can also be used to dynamically specify the length of a return
    sequence from a node. Here's an example where the length of the sequence is determined
    by the :class:`Len` instance during processing.

    .. code-block:: python

        match_length = Len()

        class CustomConfig(BaseDataProcessorConfig):
            value: int

        class CustomProcessor(BaseDataProcessor[CustomConfig]):

            def process(
                self,
                ctx: RunContext,
                a: Annotated[Sequence[Int], match_length]
            ) -> Annotated[Sequence[Float], match_length]:
                # processing logic
                ...

    In this example, the length of the sequence :code:`a` is dynamically captured by the
    :code:`match_length` instance, and the same :code:`match_length` is applied to the
    return sequence.

    **Strict Mode**:

    When the :code:`strict=True` argument is passed, the behavior of :class:`Len` changes:

    - The length of the sequence must be explicitly defined when creating the :class:`Len`
      instance (e.g., :code:`Len(4)`).
    - The class **only checks that the sequence length matches the expected length**, and
      all logic for setting or capturing the length is disabled.
    - No dynamic length setting or matching logic occurs in strict mode; it is purely for
      verifying that the sequence length is exactly as expected.

    Example: **Using Strict Mode to Enforce a Specific Sequence Length**

    .. code-block:: python

        features = datasets.Features()
        features["seq"] = datasets.Sequence(datasets.Value("Int32"), length=4)

        class Inputs(Mapping):
            seq: Annotated[Sequence[Int], Len(4, strict=True)]

        flow = DataFlow[Inputs](features)

    In this case, :code:`Len(4, strict=True)` will ensure that :code:`seq` must have exactly 4
    elements. No other logic will be applied, and no length will be dynamically set or captured.

    **Behavior**:

    - If the length of a sequence is not defined beforehand, the :class:`Len` instance can
      specify the length in non-strict mode.
    - If the sequence's length is predefined, the :class:`Len` instance checks that the
      length matches the predefined value.
    - When used across multiple sequences, the :class:`Len` instance ensures the sequences have
      the same length in non-strict mode.
    - In strict mode, the expected length must be explicitly defined, and only length validation
      occurs.
    """

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, length: int) -> None:
        ...

    @overload
    def __init__(self, length: LengthComputeFn) -> None:
        ...

    @overload
    def __init__(self, length: None | int | LengthComputeFn, strict: bool) -> None:
        ...

    @overload
    def __init__(self, length: int | LengthComputeFn, strict: Literal[True]) -> None:
        ...

    def __init__(self, length: None | int | LengthComputeFn = None, strict: bool = False) -> None:
        """Initializes the :class:`Len` validator with the expected length of the sequence.

        Args:
            length (int | None): The expected length of the sequence to be validated.
            strict (bool): Determines the validation behavior when the length is undefined.
                - If :code:`strict` is :code:`True`, the validator raises an exception if the
                  length of the sequence is undefined or does not match the expected length.
                - If :code:`strict` is :code:`False`, the validator allows undefined sequence
                  lengths and sets the length of sequences with undefined length to the expected
                  length.
        """
        # length must be set in strict mode
        assert (length is not None) if strict else True

        def length_validator(
            seq: SequenceFeature, config: BaseConfig, session: ValidationSession
        ) -> Any:
            assert isinstance(seq, SequenceFeature), (
                "The provided value is not a SequenceFeature. Ensure the correct "
                "usage of the length validator."
            )

            # get the expected length, potentially from the session context
            actual_length = len(seq.dtype)
            captured_length: CapturedLength = session.get_context(self)
            if isinstance(length, Callable):
                expected_length = length(config, captured_length, session)
            elif length is not None:
                expected_length = length
            else:
                expected_length = captured_length

            if strict or (
                (actual_length != UNDEFINED_SEQUENCE_LENGTH) and (expected_length is not None)
            ):
                # sequence feature length and expected length are both well defined
                # so check if they match up
                if expected_length != actual_length:
                    raise PydanticCustomError(
                        "Sequence Length Mismatch",
                        (
                            "Sequence length mismatch: expected {expected_length}, "
                            "but got {actual_length}."
                        ),
                        {"expected_length": expected_length, "actual_length": actual_length},
                    )

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
