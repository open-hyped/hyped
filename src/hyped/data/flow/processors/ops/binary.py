"""Binary operations.

This module defines a framework for binary operations, including mathematical, 
logical, and comparator operations. It provides base classes and configurations 
for validating input features, inferring output features, and processing binary 
operations on sequences or single values.

Key components include:
- :class:`BaseBinaryOp` and derived classes for specific operations (e.g., :class:`Add`, :class:`Equals`).
- Configuration classes for setting up operations (e.g., :class:`BaseBinaryOpConfig`).
- Input and output reference classes (e.g., :class:`BinaryOpInputRefs`, :class:`BinaryOpOutputRefs`) 
  to handle feature types and enforce constraints.
"""
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import Annotated, Any, TypeVar

from datasets import Sequence, Value
from datasets.features.features import FeatureType
from typing_extensions import Unpack

from hyped.common.feature_checks import (
    NUMERIC_TYPES,
    check_feature_is_sequence,
    get_sequence_feature,
    get_sequence_length,
    raise_sequence_lengths_match,
)
from hyped.data.flow.core.nodes.base import IOContext
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.flow.core.refs.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    GlobalValidator,
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class BaseBinaryOpConfig(BaseDataProcessorConfig):
    """Configuration class for binary operations."""


def validate_binary_inputs(
    config: BaseBinaryOpConfig, refs: BinaryOpInputRefs
) -> None:
    """Validate the input features for binary operations.

    Ensures that if both inputs are sequences of fixed length, the lengths should match.

    Args:
        config (BaseBinaryOpConfig): The configuration for the binary element-wise
            operation.
        refs (BinaryOpInputRefs): The input references containing the features to be
            validated.

    Raises:
        exc (TypeError): If the lengths of sequence features with fixed length
            do not match.
    """
    # get the features
    a_feat = refs["a"].feature_
    b_feat = refs["b"].feature_
    # get the feature names (only for error messages)
    a_key = refs["a"].key_
    b_key = refs["b"].key_

    if check_feature_is_sequence(a_feat) and check_feature_is_sequence(b_feat):
        # get the sequence lengths
        a_length = get_sequence_length(a_feat)
        b_length = get_sequence_length(b_feat)
        # check if both sequences have fixed length
        if a_length != -1 and b_length != -1:
            # raise an error if both sequences have a fixed length, but do not match
            raise_sequence_lengths_match(a_key, b_key, a_feat, b_feat)


class BinaryOpInputRefs(
    Annotated[InputRefs, GlobalValidator(validate_binary_inputs)]
):
    """Input references for binary operations."""

    a: Annotated[
        FeatureRef, CheckFeatureIsSequence(Value) | CheckFeatureEquals(Value)
    ]
    """The first input feature. Must be a sequence or value."""

    b: Annotated[
        FeatureRef, CheckFeatureIsSequence(Value) | CheckFeatureEquals(Value)
    ]
    """The second input feature. Must be a sequence or value."""


class BinaryOpNumericInputRefs(BinaryOpInputRefs):
    """Defines input references of numeric type for binary operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(NUMERIC_TYPES)
        | CheckFeatureEquals(NUMERIC_TYPES),
    ]
    """The first input feature. Must be a sequence or value of numeric type."""

    b: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(NUMERIC_TYPES)
        | CheckFeatureEquals(NUMERIC_TYPES),
    ]
    """The second input feature. Must be a sequence or value of numeric type."""


class BinaryOpBooleanInputRefs(BinaryOpInputRefs):
    """Defines input references of boolean type for binary operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool"))
        | CheckFeatureEquals(Value("bool")),
    ]
    """The first input feature. Must be a sequence or value of boolean type."""

    b: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool"))
        | CheckFeatureEquals(Value("bool")),
    ]
    """The second input feature. Must be a sequence or value of boolean type."""


NUMERIC_TYPE_NAMES = [t.dtype for t in NUMERIC_TYPES]


def binary_op_infer_output_feature(
    config: BaseBinaryOpConfig,
    inputs: BinaryOpInputRefs,
    override: str | None = None,
) -> FeatureType:
    """Infer the output feature type for a binary operation.

    This function determines the appropriate output feature type for a binary operation
    based on the input features. If both inputs are sequences, the output will be a
    sequence with a length derived from the shorter input sequence. If one input is a
    sequence and the other is not, the output will be a sequence with the same length
    as the sequence input. The data type of the output feature is determined by comparing
    the data types of the inputs, or using the `override` parameter if provided.

    Args:
        config (BaseBinaryOpConfig): The configuration for the binary operation.
        inputs (BinaryOpInputRefs): The input references containing the features to be
            processed.
        override (str | None, optional): A string representing the feature type to
            override the inferred type. If not provided, the feature type is inferred
            from the input features.

    Returns:
        FeatureType: The inferred output feature type, which may be a sequence with a
        specified inner type and length, or a single feature type if both inputs are
        non-sequences.
    """
    # get the input features
    a_feat = inputs["a"].feature_
    b_feat = inputs["b"].feature_

    if check_feature_is_sequence(a_feat) and check_feature_is_sequence(b_feat):
        # if one of a, b has length -1,
        # we want to use that as our output length
        length = min(get_sequence_length(a_feat), get_sequence_length(b_feat))
        a_type = get_sequence_feature(a_feat).dtype
        b_type = get_sequence_feature(b_feat).dtype
        feat = override or Value(
            max(a_type, b_type, key=NUMERIC_TYPE_NAMES.index)
        )
        return Sequence(feat, length=length)

    elif check_feature_is_sequence(a_feat) and not check_feature_is_sequence(
        b_feat
    ):
        length = get_sequence_length(a_feat)
        a_type = get_sequence_feature(a_feat).dtype
        feat = override or Value(
            max(a_type, b_feat.dtype, key=NUMERIC_TYPE_NAMES.index)
        )
        return Sequence(feat, length=length)

    elif not check_feature_is_sequence(a_feat) and check_feature_is_sequence(
        b_feat
    ):
        length = get_sequence_length(b_feat)
        b_type = get_sequence_feature(b_feat).dtype
        feat = override or Value(
            max(a_feat.dtype, b_type, key=NUMERIC_TYPE_NAMES.index)
        )
        return Sequence(feat, length=length)

    else:
        return override or Value(
            max(a_feat.dtype, b_feat.dtype, key=NUMERIC_TYPE_NAMES.index)
        )


class BinaryOpOutputRefs(OutputRefs):
    """Defines output references for binary operations."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(binary_op_infer_output_feature)
    ]
    """The result of the operation. Type is inferred from the input features."""


class BinaryOpIntOutputRefs(BinaryOpOutputRefs):
    """Defines output references for binary operations with integer result."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: binary_op_infer_output_feature(
                c, i, override=Value("int64")
            )
        ),
    ]
    """The integer result of the operation. If sequence inputs, this will
    be a sequence of integer."""


class BinaryOpFloatOutputRefs(BinaryOpOutputRefs):
    """Defines output references for binary operations with float result."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: binary_op_infer_output_feature(
                c, i, override=Value("float64")
            )
        ),
    ]
    """The float result of the operation. If sequence inputs, this will
    be a sequence of float."""


class BinaryOpBooleanOutputRefs(BinaryOpOutputRefs):
    """Defines output references for binary operations with boolean result."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: binary_op_infer_output_feature(
                c, i, override=Value("bool")
            )
        ),
    ]
    """The boolean result of the operation. If sequence inputs, this will
    be a sequence of boolean."""


C = TypeVar("C", bound=BaseBinaryOpConfig)
I = TypeVar("I", bound=BinaryOpInputRefs)
O = TypeVar("O", bound=BinaryOpOutputRefs)


class BaseBinaryOp(BaseDataProcessor[C, I, O], ABC):
    """Base class for binary operations."""

    @abstractmethod
    def op(self, a: Any, b: Any) -> Any:
        """The binary operation to apply."""

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Processes a batch of inputs, applying the binary operation.

        Args:
            inputs (Batch): The input batch containing sequence feature 'a'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            batch (Batch): The batch containing the result of the binary operation.
        """
        # check if one of the inputs is a sequence
        a_is_sequence = check_feature_is_sequence(io.inputs["a"])
        b_is_sequence = check_feature_is_sequence(io.inputs["b"])

        # check for broadcasting
        if a_is_sequence and b_is_sequence:
            iterator = (
                [self.op(a, b) for a, b in zip(seq_a, seq_b)]
                for seq_a, seq_b in zip(inputs["a"], inputs["b"])
            )
        elif a_is_sequence and (not b_is_sequence):
            iterator = (
                [self.op(a, b) for a in seq_a]
                for seq_a, b in zip(inputs["a"], inputs["b"])
            )
        elif (not a_is_sequence) and b_is_sequence:
            iterator = (
                [self.op(a, b) for b in seq_b]
                for a, seq_b in zip(inputs["a"], inputs["b"])
            )
        else:
            iterator = (
                self.op(a, b) for a, b in zip(inputs["a"], inputs["b"])
            )

        return {"result": list(iterator)}

    def call(self, **kwargs: Unpack[BinaryOpInputRefs]) -> O:
        """Add the binary op node to the data flow.

        This method processes the input references for the binary operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            a (FeatureRef): The first input feature.
            b (FeatureRef): The second input feature.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            BinaryOpOutputRefs: The output references produced by the
                binary operation processor.
        """
        return super(BaseBinaryOp, self).call(**kwargs)


class BaseComparatorConfig(BaseBinaryOpConfig):
    """Configuration class for comparator operations."""


C_ = TypeVar("C_", bound=BaseComparatorConfig)


class BaseComparator(BaseBinaryOp[C_, I, BinaryOpBooleanOutputRefs]):
    """Base class for comparator operations.

    Comparators are characterized by their ability to take inputs of
    any type and output a boolean sequence feature.
    """

    @abstractmethod
    def op(self, a: Any, b: Any) -> bool:
        """The comparator operation to be applied."""


class EqualsConfig(BaseComparatorConfig):
    """Configuration class for the equality operation."""


class Equals(BaseComparator[EqualsConfig, BinaryOpInputRefs]):
    """Processor for the equality operation."""

    op = operator.eq


class NotEqualsConfig(BaseComparatorConfig):
    """Configuration class for the inequality operation."""


class NotEquals(BaseComparator[NotEqualsConfig, BinaryOpInputRefs]):
    """Processor for the inequality operation."""

    op = operator.ne


class LessThanConfig(BaseComparatorConfig):
    """Configuration class for the less-than operation."""


class LessThan(BaseComparator[LessThanConfig, BinaryOpNumericInputRefs]):
    """Processor for the less-than operation."""

    op = operator.lt


class LessThanOrEqualConfig(BaseComparatorConfig):
    """Configuration class for the less-than-or-equal operation."""


class LessThanOrEqual(
    BaseComparator[LessThanOrEqualConfig, BinaryOpNumericInputRefs]
):
    """Processor for the less-than-or-equal operation."""

    op = operator.le


class GreaterThanConfig(BaseComparatorConfig):
    """Configuration class for the greater-than operation."""


class GreaterThan(BaseComparator[GreaterThanConfig, BinaryOpNumericInputRefs]):
    """Processor for the greater-than operation."""

    op = operator.gt


class GreaterThanOrEqualConfig(BaseComparatorConfig):
    """Configuration class for the greater-than-or-equal operation."""


class GreaterThanOrEqual(
    BaseComparator[GreaterThanOrEqualConfig, BinaryOpNumericInputRefs]
):
    """Processor for the greater-than-or-equal operation."""

    op = operator.ge


class BaseLogicalOpConfig(BaseComparatorConfig):
    """Configuration class for logical operations."""


C__ = TypeVar("C__", bound=BaseLogicalOpConfig)


class BaseLogicalOp(BaseComparator[C__, BinaryOpBooleanInputRefs]):
    """Base class for logical operations."""

    @abstractmethod
    def op(self, a: bool, b: bool) -> bool:
        """The logical operation to be applied."""


class LogicalAndConfig(BaseLogicalOpConfig):
    """Configuration class for the logical AND operation."""


class LogicalAnd(BaseLogicalOp[LogicalAndConfig]):
    """Processor for the logical AND operation."""

    op = operator.and_


class LogicalOrConfig(BaseLogicalOpConfig):
    """Configuration class for the logical OR operation."""


class LogicalOr(BaseLogicalOp[LogicalOrConfig]):
    """Processor for the logical OR operation."""

    op = operator.or_


class LogicalXOrConfig(BaseLogicalOpConfig):
    """Configuration class for the logical XOR operation."""


class LogicalXOr(BaseLogicalOp[LogicalXOrConfig]):
    """Processor for the logical XOR operation."""

    op = operator.xor


class BaseClosedOpConfig(BaseBinaryOpConfig):
    """Configuration class for closed mathematical operations."""


C___ = TypeVar("C___", bound=BaseBinaryOpConfig)


class BaseClosedOp(BaseBinaryOp[C___, BinaryOpNumericInputRefs, O]):
    """Base class for closed mathematical operations.

    Closed operations are characterized by preserving closure within the set of
    values they operate on. Such operators include addition or subtraction, but
    not division, as division may result in values outside the set of integers or
    floats when dividing certain numbers. For example, while adding two integers
    always results in another integer, dividing one integer by another may produce
    a non-integer result.
    """

    @abstractmethod
    def op(self, a: int | float, b: int | float) -> int | float:
        """The closed mathematical operation to be applied.

        Closed mathematical operations are operations where applying the operation to two operands
        always results in a value within the same data type domain as the operands, preserving
        closure within the set of values.
        """


class AddConfig(BaseClosedOpConfig):
    """Configuration class for the addition operation."""


class Add(BaseClosedOp[AddConfig, BinaryOpOutputRefs]):
    """Processor for the addition operation."""

    op = operator.add


class SubConfig(BaseClosedOpConfig):
    """Configuration class for the subtraction operation."""


class Sub(BaseClosedOp[SubConfig, BinaryOpOutputRefs]):
    """Processor for the subtraction operation."""

    op = operator.sub


class MulConfig(BaseClosedOpConfig):
    """Configuration class for the multiplication operation."""


class Mul(BaseClosedOp[MulConfig, BinaryOpOutputRefs]):
    """Processor for the multiplication operation."""

    op = operator.mul


class PowConfig(BaseClosedOpConfig):
    """Configuration class for the power operation."""


class Pow(BaseClosedOp[PowConfig, BinaryOpOutputRefs]):
    """Processor for the power operation."""

    op = operator.pow


class ModConfig(BaseClosedOpConfig):
    """Configuration class for the modulus operation."""


class Mod(BaseClosedOp[ModConfig, BinaryOpOutputRefs]):
    """Processor for the modulus operation."""

    op = operator.mod


class FloorDivConfig(BaseClosedOpConfig):
    """Configuration class for the floor division operation."""


class FloorDiv(BaseClosedOp[FloorDivConfig, BinaryOpIntOutputRefs]):
    """Processor for the floor division operation."""

    op = operator.floordiv


class TrueDivConfig(BaseClosedOpConfig):
    """Configuration class for the true division operation."""


class TrueDiv(BaseClosedOp[TrueDivConfig, BinaryOpFloatOutputRefs]):
    """Processor for the true division operation."""

    op = operator.truediv
