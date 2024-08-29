"""Module for performing element-wise binary operations on sequences.

This module defines a comprehensive set of processors for element-wise binary operations, 
including arithmetic, logical, and comparison operations. These processors apply binary 
operations to corresponding elements in two sequences or a sequence and a scalar.

Key operations covered by this module include addition, subtraction, multiplication, 
division, comparison (equality, inequality, greater than, less than), and logical 
operations (AND, OR, XOR). These operations can handle sequences of numeric, integer, 
and boolean types, ensuring that the data types of inputs and outputs are validated 
and consistent.

Each processor class extends the base class :class:`BaseBinaryElementWiseOp`, and 
operations are defined by overriding the `op` method, which performs the specific 
binary operation. The input references (:class:`BinaryElementWiseOpNumericInputRefs`, 
:class:`BinaryElementWiseOpBooleanInputRefs`) ensure that the inputs meet the required 
type constraints, either as sequences or individual scalar values. Output references 
(:class:`ElementWiseClosedOpOutputRefs`, :class:`BinaryElementWiseOpBooleanOutputRefs`) 
specify the type and structure of the operation's result.
"""
import operator
from abc import abstractmethod
from typing import Annotated, Any, TypeVar

from datasets import Sequence, Value

from hyped.common.feature_checks import (
    NUMERIC_TYPES,
    check_feature_is_sequence,
    get_sequence_feature,
    raise_value_feature_is_castable,
)
from hyped.data.flow.core.refs.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    GlobalValidator,
)
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops.base import BaseBinaryOpOutputRefs
from hyped.data.flow.processors.ops.sequence.element_wise.base import (
    BaseBinaryElementWiseOp,
    BaseBinaryElementWiseOpConfig,
    BinaryElementWiseOpBooleanOutputRefs,
    BinaryElementWiseOpInputRefs,
    infer_output_length,
)


def validate_sequence_castable(
    config: BaseBinaryElementWiseOpConfig,
    refs: "BinaryElementWiseOpNumericInputRefs",
) -> None:
    """Validate that features in a binary element-wise operation are castable.

    This function checks whether the input features are sequences or scalar
    values and ensures that they can be cast to each other. If both features
    are sequences, it validates that their underlying value types are castable.
    If only one feature is a sequence, it checks that the sequence's element
    type is castable to the other feature's type. If neither feature is a
    sequence, a `TypeError` is raised as at least one sequence type is required
    for the operation.

    Arguments:
        config (BaseBinaryElementWiseOpConfig): The configuration for the binary element-wise operation.
        refs (BinaryElementWiseOpNumericInputRefs): The input references containing the features to be
        validated.

    Raises:
        TypeError: If both features are of `Value` type without at least one being a sequence, or if the
        sequence features are not castable to each other.
    """
    a_key = refs["a"].key_
    b_key = refs["b"].key_

    a_feat = refs["a"].feature_
    b_feat = refs["b"].feature_

    if check_feature_is_sequence(a_feat) and check_feature_is_sequence(b_feat):
        raise_value_feature_is_castable(
            a_key,
            b_key,
            get_sequence_feature(a_feat),
            get_sequence_feature(b_feat),
        )

    elif check_feature_is_sequence(a_feat) and not check_feature_is_sequence(
        b_feat
    ):
        raise_value_feature_is_castable(
            a_key,
            b_key,
            get_sequence_feature(a_feat),
            b_feat,
        )

    elif not check_feature_is_sequence(a_feat) and check_feature_is_sequence(
        b_feat
    ):
        raise_value_feature_is_castable(
            a_key,
            b_key,
            a_feat,
            get_sequence_feature(b_feat),
        )

    else:
        raise TypeError(
            "Both features for element-wise binary operation are of Value type. "
            f"Need min. 1 sequence type. Got {a_key} ({a_feat}) and {b_key} ({b_feat})"
        )


class BinaryElementWiseOpNumericInputRefs(
    Annotated[
        BinaryElementWiseOpInputRefs,
        GlobalValidator(validate_sequence_castable),
    ]
):
    """Defines input references for binary element-wise operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(NUMERIC_TYPES)
        | CheckFeatureEquals(NUMERIC_TYPES),
    ]
    """The first input feature. Must be a numeric value or sequence of numerical type."""

    b: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(NUMERIC_TYPES)
        | CheckFeatureEquals(NUMERIC_TYPES),
    ]
    """The second input feature. Must be a numeric value or sequence of numerical type."""


class BinaryElementWiseOpBooleanInputRefs(BinaryElementWiseOpInputRefs):
    """Defines input references for boolean element-wise binary operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool"))
        | CheckFeatureEquals(Value("bool")),
    ]
    """The first input feature. Must be a boolean value or sequence of boolean."""

    b: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool"))
        | CheckFeatureEquals(Value("bool")),
    ]
    """The second input feature. Must be a boolean value or sequence of boolean."""


class BaseElementWiseComparatorConfig(BaseBinaryElementWiseOpConfig):
    """Configuration class for element-wise comparator operations."""


C = TypeVar("C", bound=BaseElementWiseComparatorConfig)
I = TypeVar("I", bound=BinaryElementWiseOpInputRefs)


class BaseElementWiseComparator(
    BaseBinaryElementWiseOp[C, I, BinaryElementWiseOpBooleanOutputRefs]
):
    """Base class for element-wise comparator operations.

    Comparators are characterized by their ability to take inputs of non-nested sequences
    and output a boolean sequence feature.
    """

    @abstractmethod
    def op(self, a: Any, b: Any) -> bool:
        """The comparator operation to be applied.

        Takes sequence inputs and outputs a boolean sequence feature.
        """


class ElementWiseEqualsConfig(BaseElementWiseComparatorConfig):
    """Configuration class for the equality operation."""


class ElementWiseEquals(
    BaseElementWiseComparator[
        ElementWiseEqualsConfig, BinaryElementWiseOpInputRefs
    ]
):
    """Processor for the equality operation."""

    op = operator.eq


class ElementWiseNotEqualsConfig(BaseElementWiseComparatorConfig):
    """Configuration class for the inequality operation."""


class ElementWiseNotEquals(
    BaseElementWiseComparator[
        ElementWiseNotEqualsConfig, BinaryElementWiseOpInputRefs
    ]
):
    """Processor for the inequality operation."""

    op = operator.ne


class ElementWiseLessThanConfig(BaseElementWiseComparatorConfig):
    """Configuration class for the less-than operation."""


class ElementWiseLessThan(
    BaseElementWiseComparator[
        ElementWiseLessThanConfig, BinaryElementWiseOpNumericInputRefs
    ]
):
    """Processor for the less-than operation."""

    op = operator.lt


class ElementWiseLessThanOrEqualConfig(BaseElementWiseComparatorConfig):
    """Configuration class for the less-than-or-equal operation."""


class ElementWiseLessThanOrEqual(
    BaseElementWiseComparator[
        ElementWiseLessThanOrEqualConfig, BinaryElementWiseOpNumericInputRefs
    ]
):
    """Processor for the less-than-or-equal operation."""

    op = operator.le


class ElementWiseGreaterThanConfig(BaseElementWiseComparatorConfig):
    """Configuration class for the greater-than operation."""


class ElementWiseGreaterThan(
    BaseElementWiseComparator[
        ElementWiseGreaterThanConfig, BinaryElementWiseOpNumericInputRefs
    ]
):
    """Processor for the greater-than operation."""

    op = operator.gt


class ElementWiseGreaterThanOrEqualConfig(BaseElementWiseComparatorConfig):
    """Configuration class for the greater-than-or-equal operation."""


class ElementWiseGreaterThanOrEqual(
    BaseElementWiseComparator[
        ElementWiseGreaterThanOrEqualConfig,
        BinaryElementWiseOpNumericInputRefs,
    ]
):
    """Processor for the greater-than-or-equal operation."""

    op = operator.ge


class BaseElementWiseLogicalOpConfig(BaseElementWiseComparatorConfig):
    """Configuration class for logical operations."""


C = TypeVar("C", bound=BaseElementWiseLogicalOpConfig)


class BaseElementWiseLogicalOp(
    BaseElementWiseComparator[C, BinaryElementWiseOpBooleanInputRefs]
):
    """Base class for logical operations."""

    @abstractmethod
    def op(self, a: bool, b: bool) -> bool:
        """The logical operation to be applied.

        Takes boolean inputs and outputs a boolean feature.
        """


class ElementWiseLogicalAndConfig(BaseElementWiseLogicalOpConfig):
    """Configuration class for the logical AND operation."""


class ElementWiseLogicalAnd(
    BaseElementWiseLogicalOp[ElementWiseLogicalAndConfig]
):
    """Processor for the logical AND operation."""

    op = operator.and_


class ElementWiseLogicalOrConfig(BaseElementWiseLogicalOpConfig):
    """Configuration class for the logical OR operation."""


class ElementWiseLogicalOr(
    BaseElementWiseLogicalOp[ElementWiseLogicalOrConfig]
):
    """Processor for the logical OR operation."""

    op = operator.or_


class ElementWiseLogicalXOrConfig(BaseElementWiseLogicalOpConfig):
    """Configuration class for the logical XOR operation."""


class ElementWiseLogicalXOr(
    BaseElementWiseLogicalOp[ElementWiseLogicalXOrConfig]
):
    """Processor for the logical XOR operation."""

    op = operator.xor


class BaseElementWiseClosedOpConfig(BaseBinaryElementWiseOpConfig):
    """Configuration class for closed mathematical operations."""


NUMERICAL_TYPE_NAMES = [t.dtype for t in NUMERIC_TYPES]


def element_wise_closed_op_infer_dtype(
    config: BaseElementWiseClosedOpConfig,
    inputs: BinaryElementWiseOpNumericInputRefs,
) -> Value:
    """Infers the output data type for closed operations based on input types.

    For closed operations, the inferred output data type is determined by the input data types.
    If both inputs are integers, the output data type will be an integer. If both inputs are floats,
    the output data type will be a float. For a mixture of integers and floats, the output data
    type will be float.

    Args:
        config (ClosedOpConfig): The configuration for the closed operation.
        inputs (MathInputRefs): The input references.

    Returns:
        Value: The inferred data type for the output.
    """
    if check_feature_is_sequence(inputs["a"].feature_):
        a_type = get_sequence_feature(inputs["a"].feature_).dtype
    else:
        a_type = inputs["a"].feature_.dtype

    if check_feature_is_sequence(inputs["b"].feature_):
        b_type = get_sequence_feature(inputs["b"].feature_).dtype
    else:
        b_type = inputs["b"].feature_.dtype

    return Sequence(
        Value(
            # this order prefers higher precision types
            max(a_type, b_type, key=NUMERICAL_TYPE_NAMES.index)
        ),
        length=infer_output_length(inputs),
    )


class ElementWiseClosedOpOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for closed mathematical operations."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(element_wise_closed_op_infer_dtype)
    ]
    """The result of the closed mathematical operation."""


C = TypeVar("C", bound=BaseElementWiseClosedOpConfig)


class BaseElementWiseClosedOp(
    BaseBinaryElementWiseOp[
        C, BinaryElementWiseOpNumericInputRefs, ElementWiseClosedOpOutputRefs
    ]
):
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


class ElementWiseAddConfig(BaseElementWiseClosedOpConfig):
    """Configuration class for the addition operation."""


class ElementWiseAdd(BaseElementWiseClosedOp[ElementWiseAddConfig]):
    """Processor for the addition operation."""

    op = operator.add


class ElementWiseSubConfig(BaseElementWiseClosedOpConfig):
    """Configuration class for the subtraction operation."""


class ElementWiseSub(BaseElementWiseClosedOp[ElementWiseSubConfig]):
    """Processor for the subtraction operation."""

    op = operator.sub


class ElementWiseMulConfig(BaseElementWiseClosedOpConfig):
    """Configuration class for the multiplication operation."""


class ElementWiseMul(BaseElementWiseClosedOp[ElementWiseMulConfig]):
    """Processor for the multiplication operation."""

    op = operator.mul


class ElementWisePowConfig(BaseElementWiseClosedOpConfig):
    """Configuration class for the power operation."""


class ElementWisePow(BaseElementWiseClosedOp[ElementWisePowConfig]):
    """Processor for the power operation."""

    op = operator.pow


class ElementWiseModConfig(BaseElementWiseClosedOpConfig):
    """Configuration class for the modulus operation."""


class ElementWiseMod(BaseElementWiseClosedOp[ElementWiseModConfig]):
    """Processor for the modulus operation."""

    op = operator.mod


class ElementWiseFloorDivConfig(BaseElementWiseClosedOpConfig):
    """Configuration class for the floor division operation."""


class ElementWiseFloorDivOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for the floor division operation."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                Value("int32"), length=infer_output_length(i)
            )
        ),
    ]
    """The result of the floor division operation."""


class ElementWiseFloorDiv(
    BaseBinaryElementWiseOp[
        ElementWiseFloorDivConfig,
        BinaryElementWiseOpNumericInputRefs,
        ElementWiseFloorDivOutputRefs,
    ]
):
    """Processor for the floor division operation."""

    op = operator.floordiv


class ElementWiseTrueDivConfig(BaseElementWiseClosedOpConfig):
    """Configuration class for the true division operation."""


class ElementWiseTrueDivOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for the true division operation."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                Value("float32"), length=infer_output_length(i)
            )
        ),
    ]
    """The result of the true division operation."""


class ElementWiseTrueDiv(
    BaseBinaryElementWiseOp[
        ElementWiseTrueDivConfig,
        BinaryElementWiseOpNumericInputRefs,
        ElementWiseTrueDivOutputRefs,
    ]
):
    """Processor for the true division operation."""

    op = operator.truediv
