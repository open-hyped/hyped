"""Module containing processor implementations for sequence operators."""
import operator
from typing import Any, Callable

from datasets import Sequence
from typing_extensions import Annotated

from hyped.common.feature_checks import (
    check_feature_equals,
    get_sequence_feature,
    get_sequence_length,
)
from hyped.data.flow.core.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.core.refs.outputs import (
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.core.refs.ref import FeatureRef

from .binary import BinaryOp, BinaryOpConfig
from .unary import UnaryOp, UnaryOpConfig


class ConcatConfig(BinaryOpConfig):
    """Configuration class for the Concat operation."""

    op: Callable[[list[Any], list[Any]], list[Any]] = operator.concat
    """The concatenate operation."""


class ConcatInputRefs(InputRefs):
    """Input references for the Concat operation."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Sequence)]
    """The first sequence feature reference."""

    b: Annotated[FeatureRef, CheckFeatureEquals(Sequence)]
    """The second sequence feature reference."""

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization check to ensure sequence features align for concatenation.

        Args:
            __context: The context for the model initialization.

        Raises:
            RuntimeError: If the sequence features do not match.
        """
        # sequence features must align in
        # order to concatenate sequences
        if not check_feature_equals(
            self.a.feature_.feature,
            self.b.feature_.feature,
        ):
            raise RuntimeError(
                "The sequence features of 'a' and 'b' must match to perform concatenation."
            )


def infer_concat_output_dtype(
    config: ConcatConfig, inputs: ConcatInputRefs
) -> Sequence:
    """Infer the output data type for the Concat operation.

    Args:
        config (ConcatConfig): The configuration for the Concat operation.
        inputs (ConcatInputRefs): The input references for the Concat operation.

    Returns:
        Sequence: The output sequence feature with inferred length.
    """
    # get the lengths of the input sequences
    a_length = get_sequence_length(inputs.a.feature_)
    b_length = get_sequence_length(inputs.b.feature_)
    # compute the length of the output sequence
    length = -1 if -1 in (a_length, b_length) else a_length + b_length
    # build the output sequence feature assuming that
    # the input sequence features match
    return Sequence(
        feature=get_sequence_feature(inputs.a.feature_), length=length
    )


class ConcatOutputRefs(OutputRefs):
    """Output references for the Concat operation."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(infer_concat_output_dtype)
    ]
    """The feature reference to the result of the concatenation operation."""


class Concat(BinaryOp[ConcatConfig, ConcatInputRefs, ConcatOutputRefs]):
    """The Concat operation class, inheriting from BinaryOp.

    This class defines the concatenation operation for sequence features.
    """
