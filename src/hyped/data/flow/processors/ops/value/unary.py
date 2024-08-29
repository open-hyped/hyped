"""Module for unary operations in data processing pipelines.

This module defines various unary operations for processing individual features.
"""
import operator
from typing import Annotated

from datasets import Value

from hyped.common.feature_checks import INT_TYPES, NUMERIC_TYPES
from hyped.data.flow.core.refs.inputs import CheckFeatureEquals
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops.base import (
    BaseUnaryOp,
    BaseUnaryOpConfig,
    BaseUnaryOpOutputRefs,
    BooleanOutputRefs,
    UnaryOpInputRefs,
)


class UnaryOpNumericInputRefs(UnaryOpInputRefs):
    """Defines input references for unary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(NUMERIC_TYPES)]
    """The input feature. Must be a numeric type."""


class UnaryOpIntInputRefs(UnaryOpNumericInputRefs):
    """Defines input references for unary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(INT_TYPES)]
    """The input feature. Must be a integer type."""


class UnaryOpBooleanInputRefs(UnaryOpInputRefs):
    """Defines input references for unary operations."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Value("bool"))]
    """The input feature. Must be boolean type."""


class UnaryOpNumericOutputRefs(BaseUnaryOpOutputRefs):
    """Defines output references for numeric unary operations."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(lambda config, inputs: inputs["a"].feature_),
    ]
    """The result of the numeric unary operation."""


class NegConfig(BaseUnaryOpConfig):
    """Configuration class for the negation operation."""


class Neg(
    BaseUnaryOp[NegConfig, UnaryOpNumericInputRefs, UnaryOpNumericOutputRefs]
):
    """Processor for the negation operation."""

    op = operator.neg


class AbsConfig(BaseUnaryOpConfig):
    """Configuration class for the absolute operation."""


class Abs(
    BaseUnaryOp[AbsConfig, UnaryOpNumericInputRefs, UnaryOpNumericOutputRefs]
):
    """Processor for the absolute operation."""

    op = operator.abs


class InvertConfig(BaseUnaryOpConfig):
    """Configuration class for the bitwise inversion operation."""


class Invert(
    BaseUnaryOp[InvertConfig, UnaryOpIntInputRefs, UnaryOpNumericOutputRefs]
):
    """Processor for the bitwise inversion operation."""

    op = operator.invert


class BooleanInvertConfig(BaseUnaryOpConfig):
    """Configuration class for the boolean inversion operation."""


class BooleanInvert(
    BaseUnaryOp[
        BooleanInvertConfig, UnaryOpBooleanInputRefs, BooleanOutputRefs
    ]
):
    """Processor for the negation operation."""

    # operator.invert is not the same as boolean negation
    def op(self, b: bool) -> bool:
        """Invert the boolean value.

        Args:
            b (bool): The boolean input.

        Returns:
        (bool): The inversion.
        """
        return not b
