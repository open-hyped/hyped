"""Unary operations.

This module defines a framework for unary operations, including mathematical and
boolean operations. It provides base classes and configurations for processing 
unary operations on features, inferring output feature types, and handling 
special cases like boolean inversion.

Key components include:
- :class:`BaseUnaryOp` and its derived classes for specific unary operations 
  (e.g., :class:`Neg`, :class:`Abs`).
- :class:`BaseUnaryOpConfig` for configuring unary operations.
- Input and output reference classes like :class:`UnaryOpInputRefs` and 
  :class:`UnaryOpOutputRefs` to manage feature types and constraints.
- Utility function :code:`unary_op_infer_output_feature` for determining output 
  feature types based on inputs.
"""
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import Annotated, Any, TypeVar

from datasets import Sequence, Value
from datasets.features.features import FeatureType
from typing_extensions import Unpack

from hyped.common.feature_checks import (
    INDEX_TYPES,
    NUMERIC_TYPES,
    check_feature_is_sequence,
    get_sequence_feature,
    get_sequence_length,
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
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class BaseUnaryOpConfig(BaseDataProcessorConfig):
    """Configuration class for unary operations."""


class UnaryOpInputRefs(InputRefs):
    """Input references for unary operations."""

    a: Annotated[
        FeatureRef, CheckFeatureIsSequence(Value) | CheckFeatureEquals(Value)
    ]
    """The input feature. Must be a sequence or value."""


def unary_op_infer_output_feature(
    config: BaseUnaryOpConfig,
    inputs: UnaryOpInputRefs,
    override: str | None = None,
) -> FeatureType:
    """Infer the output feature type for a unary operation.

    This function determines the appropriate output feature type for a unary operation
    based on the input feature type. If the input feature is a sequence, the output
    will be a sequence with the same length but potentially a different inner feature
    type, as specified by the `override` parameter. If the input is not a sequence, the
    output will match the input type or the `override` type if provided.

    Args:
        config (BaseUnaryOpConfig): The configuration for the unary operation.
        inputs (UnaryOpInputRefs): The input references containing the feature to be
            processed.
        override (str | None, optional): A string representing the feature type to
            override the inferred type. If not provided, the feature type is inferred
            from the input.

    Returns:
        FeatureType: The inferred output feature type, either as a sequence with a
        specified inner type and length, or as a single feature type.
    """
    if check_feature_is_sequence(inputs["a"].feature_):
        feat = override or get_sequence_feature(inputs["a"].feature_)
        length = get_sequence_length(inputs["a"].feature_)
        return Sequence(feat, length=length)
    else:
        return override or inputs["a"].feature_


class UnaryOpNumericInputRefs(UnaryOpInputRefs):
    """Defines input references of numeric type for unary operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(NUMERIC_TYPES)
        | CheckFeatureEquals(NUMERIC_TYPES),
    ]
    """The input feature. Must be a sequence or value of numeric type."""


class UnaryOpIntegerInputRefs(UnaryOpInputRefs):
    """Defines input references of integer type for unary operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(INDEX_TYPES) | CheckFeatureEquals(INDEX_TYPES),
    ]
    """The input feature. Must be a sequence or value of integer type."""


class UnaryOpBooleanInputRefs(UnaryOpInputRefs):
    """Defines input references of boolean type for unary operations."""

    a: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool"))
        | CheckFeatureEquals(Value("bool")),
    ]
    """The input feature. Must be a sequence or value of boolean type."""


class UnaryOpOutputRefs(OutputRefs):
    """Defines output references for unary operations."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(unary_op_infer_output_feature)
    ]
    """The result of the operation. Type is inferred from the input feature."""


class UnaryOpBooleanOutputRefs(UnaryOpOutputRefs):
    """Defines output references for unary operations with boolean result."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: unary_op_infer_output_feature(
                c, i, override=Value("bool")
            )
        ),
    ]
    """The boolean result of the operation. If sequence inputs, this will
    be a sequence of boolean."""


C = TypeVar("C", bound=BaseUnaryOpConfig)
I = TypeVar("I", bound=UnaryOpInputRefs)
O = TypeVar("O", bound=UnaryOpOutputRefs)


class BaseUnaryOp(BaseDataProcessor[C, I, O], ABC):
    """Base class for unary operations."""

    @abstractmethod
    def op(self, val: Any) -> Any:
        """The operation to apply."""
        ...

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Processes a batch of inputs, applying the unary operation.

        Args:
            inputs (Batch): The input batch containing sequence feature 'a'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            batch (Batch): The batch containing the result of the unary operation.
        """
        if check_feature_is_sequence(io.inputs["a"]):
            return {
                "result": [[self.op(a) for a in seq] for seq in inputs["a"]]
            }
        else:
            return {"result": [self.op(a) for a in inputs["a"]]}

    def call(self, **kwargs: Unpack[UnaryOpInputRefs]) -> O:
        """Add the unary op node to the data flow.

        This method processes the input references for the unary operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            a (FeatureRef): The input sequence feature for the unary operation.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            UnaryOpOutputRefs: The output references produced by the
                unary operation processor.
        """
        return super(BaseUnaryOp, self).call(**kwargs)


class NegConfig(BaseUnaryOpConfig):
    """Configuration class for the negation operation."""


class Neg(
    BaseUnaryOp[
        NegConfig,
        UnaryOpNumericInputRefs,
        UnaryOpOutputRefs,
    ]
):
    """Processor for the negation operation."""

    op = operator.neg


class AbsConfig(BaseUnaryOpConfig):
    """Configuration class for the absolute operation."""


class Abs(
    BaseUnaryOp[
        AbsConfig,
        UnaryOpNumericInputRefs,
        UnaryOpOutputRefs,
    ]
):
    """Processor for the absolute operation."""

    op = operator.abs


class InvertConfig(BaseUnaryOpConfig):
    """Configuration class for the bitwise inversion operation."""


class Invert(
    BaseUnaryOp[
        InvertConfig,
        UnaryOpIntegerInputRefs,
        UnaryOpOutputRefs,
    ]
):
    """Processor for the bitwise inversion operation."""

    op = operator.invert


class BooleanInvertConfig(BaseUnaryOpConfig):
    """Configuration class for the boolean inversion operation."""


class BooleanInvert(
    BaseUnaryOp[
        BooleanInvertConfig,
        UnaryOpBooleanInputRefs,
        UnaryOpBooleanOutputRefs,
    ]
):
    """Processor for the boolean inversion operation.

    This processor exists because booleans are a special case,
    since `operator.invert` and `operator.neg` behave not the same as `not`.
    """

    def op(self, b: bool) -> bool:
        """The invert operation.

        Args:
            b (bool): the boolean value.

        Return:
            (bool): The inversion.
        """
        return not b
