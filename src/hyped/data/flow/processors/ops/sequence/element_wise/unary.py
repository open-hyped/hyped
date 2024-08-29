"""Module for performing unary element-wise operations on sequences.

This module defines a set of processors (:class:`BaseUnaryElementWiseOp` and its subclasses) 
that apply unary operations to each element within a sequence. The operations include 
negation, absolute value, bitwise inversion, and boolean inversion.

Each processor is configured to handle a specific type of input feature, such as numeric, 
integer, or boolean sequences. The processors are designed to be integrated into data processing 
pipelines, where they can be applied to transform data on a per-element basis.

The input features for these operations are defined using specific input reference classes 
(:class:`UnaryElementWiseOpNumericInputRefs`, :class:`UnaryElementWiseOpIntInputRefs`, 
and :class:`UnaryElementWiseOpBooleanInputRefs`), which enforce the required feature types 
through type checks.

The resulting output of these operations is specified in output reference classes 
(:class:`UnaryElementWiseOpNumericOutputRefs` and :class:`UnaryElementWiseOpBooleanOutputRefs`), 
ensuring that the transformed features maintain consistency in type and structure.

This module is particularly useful for tasks requiring element-wise transformations within 
sequences, such as flipping bits, negating numbers, or applying boolean logic across elements 
in a sequence.
"""
import operator
from typing import Annotated

from datasets import Value

from hyped.common.feature_checks import INT_TYPES, NUMERIC_TYPES
from hyped.data.flow.core.refs.inputs import CheckFeatureIsSequence
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops.base import BaseUnaryOpOutputRefs
from hyped.data.flow.processors.ops.sequence.element_wise.base import (
    BaseUnaryElementWiseOp,
    BaseUnaryElementWiseOpConfig,
    UnaryElementWiseOpBooleanOutputRefs,
    UnaryElementWiseOpInputRefs,
)


class UnaryElementWiseOpNumericInputRefs(UnaryElementWiseOpInputRefs):
    """Defines input references for unary element-wise operations."""

    a: Annotated[FeatureRef, CheckFeatureIsSequence(NUMERIC_TYPES)]
    """The input feature. Must be a sequence of numeric type."""


class UnaryElementWiseOpIntInputRefs(UnaryElementWiseOpInputRefs):
    """Defines input references for unary element-wise operations."""

    a: Annotated[FeatureRef, CheckFeatureIsSequence(INT_TYPES)]
    """The input feature. Must be a sequence of integer type."""


class UnaryElementWiseOpBooleanInputRefs(UnaryElementWiseOpInputRefs):
    """Defines input references for unary element-wise operations."""

    a: Annotated[FeatureRef, CheckFeatureIsSequence(Value("bool"))]
    """The input feature. Must be sequence of boolean type."""


class UnaryElementWiseOpNumericOutputRefs(BaseUnaryOpOutputRefs):
    """Defines output references for element-wise numeric unary operations."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(lambda c, i: i["a"].feature_),
    ]
    """The result of the numeric unary operation."""


class ElementWiseNegConfig(BaseUnaryElementWiseOpConfig):
    """Configuration class for the element-wise negation operation."""


class ElementWiseNeg(
    BaseUnaryElementWiseOp[
        ElementWiseNegConfig,
        UnaryElementWiseOpNumericInputRefs,
        UnaryElementWiseOpNumericOutputRefs,
    ]
):
    """Processor for the element-wise negation operation."""

    op = operator.neg


class ElementWiseAbsConfig(BaseUnaryElementWiseOpConfig):
    """Configuration class for the absolute operation."""


class ElementWiseAbs(
    BaseUnaryElementWiseOp[
        ElementWiseAbsConfig,
        UnaryElementWiseOpNumericInputRefs,
        UnaryElementWiseOpNumericOutputRefs,
    ]
):
    """Processor for the absolute operation."""

    op = operator.abs


class ElementWiseInvertConfig(BaseUnaryElementWiseOpConfig):
    """Configuration class for the element-wise bitwise inversion operation."""


class ElementWiseInvert(
    BaseUnaryElementWiseOp[
        ElementWiseInvertConfig,
        UnaryElementWiseOpNumericInputRefs,
        UnaryElementWiseOpNumericOutputRefs,
    ]
):
    """Processor for the element-wise bitwise inversion operation."""

    op = operator.invert


class ElementWiseBooleanInvertConfig(BaseUnaryElementWiseOpConfig):
    """Configuration class for the element-wise boolean inversion operation."""


class ElementWiseBooleanInvert(
    BaseUnaryElementWiseOp[
        ElementWiseBooleanInvertConfig,
        UnaryElementWiseOpNumericInputRefs,
        UnaryElementWiseOpBooleanOutputRefs,
    ]
):
    """Processor for the element-wise boolean inversion operation."""

    # operator.invert is not the same as boolean negation
    def op(self, b: bool) -> bool:
        """The invert operation.

        Args:
            b (bool): the boolean value.
        """
        return not b
