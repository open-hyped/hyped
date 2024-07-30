import operator
from abc import abstractmethod
from typing import Annotated, Any, TypeVar

from datasets import Value

from hyped.common.feature_checks import NUMERIC_TYPES
from hyped.data.flow.core.refs.inputs import CheckFeatureEquals
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputFeature
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops.base import (
    BaseBinaryOp,
    BaseBinaryOpConfig,
    BaseBinaryOpOutputRefs,
    BinaryOpInputRefs,
    BooleanOutputRefs,
)


class BinaryOpValueInputRefs(BinaryOpInputRefs):
    """Defines input references for binary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value)]
    """The first input feature. Must be a Value type."""

    b: Annotated[FeatureRef, CheckFeatureEquals(Value)]
    """The second input feature. Must be a Value type."""


class BinaryOpNumericInputRefs(BinaryOpInputRefs):
    """Defines input references for binary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(NUMERIC_TYPES)]
    """The first input feature. Must be a numerical type."""

    b: Annotated[FeatureRef, CheckFeatureEquals(NUMERIC_TYPES)]
    """The second input feature. Must be a numerical type."""


class BinaryOpBooleanInputRefs(BinaryOpInputRefs):
    """Defines input references for boolean binary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value("bool"))]
    """The first input feature. Must be of boolean type."""

    b: Annotated[FeatureRef, CheckFeatureEquals(Value("bool"))]
    """The second input feature. Must be of boolean type."""


class BaseComparatorConfig(BaseBinaryOpConfig):
    """Configuration class for numeric comparator operations."""


C = TypeVar("C", bound=BaseComparatorConfig)
I = TypeVar("I", bound=BinaryOpInputRefs)


class BaseComparator(BaseBinaryOp[C, I, BooleanOutputRefs]):
    """Base class for numeric comparator operations.

    Comparators are characterized by their ability to take inputs of any value type
    and output a boolean feature.
    """

    @abstractmethod
    def op(self, a: Any, b: Any) -> bool:
        """The comparator operation to be applied.

        Takes numeric inputs and outputs a boolean feature.
        """


class EqualsConfig(BaseComparatorConfig):
    """Configuration class for the equality operation."""


class Equals(BaseComparator[EqualsConfig, BinaryOpValueInputRefs]):
    """Processor for the equality operation."""

    op = operator.eq


class NotEqualsConfig(BaseComparatorConfig):
    """Configuration class for the inequality operation."""


class NotEquals(BaseComparator[NotEqualsConfig, BinaryOpValueInputRefs]):
    """Processor for the inequality operation."""

    op = operator.ne


class LessThanConfig(BaseComparatorConfig):
    """Configuration class for the less-than operation."""


class LessThan(BaseComparator[LessThanConfig, BinaryOpNumericInputRefs]):
    """Processor for the less-than operation."""

    op = operator.lt


class LessThanOrEqualConfig(BaseComparatorConfig):
    """Configuration class for the less-than-or-equal operation."""


class LessThanOrEqual(BaseComparator[LessThanOrEqualConfig, BinaryOpNumericInputRefs]):
    """Processor for the less-than-or-equal operation."""

    op = operator.le


class GreaterThanConfig(BaseComparatorConfig):
    """Configuration class for the greater-than operation."""


class GreaterThan(BaseComparator[GreaterThanConfig, BinaryOpNumericInputRefs]):
    """Processor for the greater-than operation."""

    op = operator.gt


class GreaterThanOrEqualConfig(BaseComparatorConfig):
    """Configuration class for the greater-than-or-equal operation."""


class GreaterThanOrEqual(BaseComparator[GreaterThanOrEqualConfig, BinaryOpNumericInputRefs]):
    """Processor for the greater-than-or-equal operation."""

    op = operator.ge


class BaseLogicalOpConfig(BaseComparatorConfig):
    """Configuration class for logical operations."""


C = TypeVar("C", bound=BaseLogicalOpConfig)


class BaseLogicalOp(BaseComparator[C, BinaryOpBooleanInputRefs]):
    """Base class for logical operations."""

    @abstractmethod
    def op(self, a: bool, b: bool) -> bool:
        """The logical operation to be applied.

        Takes boolean inputs and outputs a boolean feature.
        """


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


NUMERICAL_TYPE_NAMES = [t.dtype for t in NUMERIC_TYPES]


def closed_op_infer_dtype(config: BaseClosedOpConfig, inputs: BinaryOpNumericInputRefs) -> Value:
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
    return Value(
        max(
            inputs["a"].feature_.dtype,
            inputs["b"].feature_.dtype,
            key=NUMERICAL_TYPE_NAMES.index,  # this order prefers higher precision types
        )
    )


class ClosedOpOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for closed mathematical operations."""

    result: Annotated[FeatureRef, LambdaOutputFeature(closed_op_infer_dtype)]
    """The result of the closed mathematical operation."""


C = TypeVar("C", bound=BaseClosedOpConfig)


class BaseClosedOp(BaseBinaryOp[C, BinaryOpNumericInputRefs, ClosedOpOutputRefs]):
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


class Add(BaseClosedOp[AddConfig]):
    """Processor for the addition operation."""

    op = operator.add


class SubConfig(BaseClosedOpConfig):
    """Configuration class for the subtraction operation."""


class Sub(BaseClosedOp[SubConfig]):
    """Processor for the subtraction operation."""

    op = operator.sub


class MulConfig(BaseClosedOpConfig):
    """Configuration class for the multiplication operation."""


class Mul(BaseClosedOp[MulConfig]):
    """Processor for the multiplication operation."""

    op = operator.mul


class PowConfig(BaseClosedOpConfig):
    """Configuration class for the power operation."""


class Pow(BaseClosedOp[PowConfig]):
    """Processor for the power operation."""

    op = operator.pow


class ModConfig(BaseClosedOpConfig):
    """Configuration class for the modulus operation."""


class Mod(BaseClosedOp[ModConfig]):
    """Processor for the modulus operation."""

    op = operator.mod


class FloorDivConfig(BaseBinaryOpConfig):
    """Configuration class for the floor division operation."""


class FloorDivOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for the floor division operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("int32"))]
    """The result of the floor division operation."""


class FloorDiv(BaseBinaryOp[FloorDivConfig, BinaryOpNumericInputRefs, FloorDivOutputRefs]):
    """Processor for the floor division operation."""

    op = operator.floordiv


class TrueDivConfig(BaseBinaryOpConfig):
    """Configuration class for the true division operation."""


class TrueDivOutputRefs(BaseBinaryOpOutputRefs):
    """Defines output references for the true division operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("float32"))]
    """The result of the true division operation."""


class TrueDiv(BaseBinaryOp[TrueDivConfig, BinaryOpNumericInputRefs, TrueDivOutputRefs]):
    """Processor for the true division operation."""

    op = operator.truediv
