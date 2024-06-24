"""Module containing processor implementations for sequence operators."""
import operator
from collections import deque
from functools import partial
from itertools import starmap
from typing import Any, Callable

import numpy as np
from datasets import Sequence, Value
from typing_extensions import Annotated

from hyped.common.feature_checks import (
    INDEX_TYPES,
    check_feature_equals,
    check_feature_is_sequence,
    get_sequence_feature,
    get_sequence_length,
)
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import (
    AnyFeatureType,
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import (
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.core.refs.ref import FeatureRef

from .binary import BinaryOp, BinaryOpConfig
from .unary import UnaryOp, UnaryOpConfig


class SequenceConcatConfig(BinaryOpConfig):
    """Configuration class for the Concat operation."""

    op: Callable[[list[Any], list[Any]], list[Any]] = operator.concat
    """The concatenate operation."""


class SequenceConcatInputRefs(InputRefs):
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
            get_sequence_feature(self.a.feature_),
            get_sequence_feature(self.b.feature_),
        ):
            raise RuntimeError(
                f"The sequence features of 'a' and 'b' must match to perform concatenation, "
                f"got `{get_sequence_feature(self.a.feature_)}` "
                f"!= `{get_sequence_feature(self.b.feature_)}`"
            )


def infer_concat_output_dtype(
    config: SequenceConcatConfig, inputs: SequenceConcatInputRefs
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


class SequenceConcatOutputRefs(OutputRefs):
    """Output references for the Concat operation."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(infer_concat_output_dtype)
    ]
    """The feature reference to the result of the concatenation operation."""


class SequenceConcat(
    BinaryOp[
        SequenceConcatConfig, SequenceConcatInputRefs, SequenceConcatOutputRefs
    ]
):
    """The Sequence Concatente Data Processor.

    This class defines the concatenation operation for sequence features.
    """


class SequenceLengthInputRefs(InputRefs):
    """Input references for the Sequence Length operation."""

    a: Annotated[FeatureRef, CheckFeatureEquals(Sequence)]
    """The sequence feature reference to get the length of."""


class SequenceLengthOutputRefs(OutputRefs):
    """Output references for the Sequence Length operation."""

    length: Annotated[FeatureRef, OutputFeature(Value("int64"))]
    """The feature reference to the length of the sequence."""


class SequenceLengthConfig(UnaryOpConfig):
    """Configuration class for the Sequence Length operation."""

    op: Callable[[list[Any]], int] = len
    """The operation to get the length of the sequence."""


class SequenceLength(
    UnaryOp[
        SequenceLengthConfig, SequenceLengthInputRefs, SequenceLengthOutputRefs
    ]
):
    """The Sequence Length Data Processor.

    This class defines the operation to get the length of a sequence feature.
    """

    CONFIG_TYPE = SequenceLengthConfig


class SequenceGetItemInputRefs(InputRefs):
    """Input references for the GetItem operation."""

    sequence: Annotated[FeatureRef, CheckFeatureEquals(Sequence)]
    """The sequence feature reference."""

    # either an index or a sequence of indices
    index: Annotated[
        FeatureRef,
        CheckFeatureEquals(INDEX_TYPES) | CheckFeatureIsSequence(INDEX_TYPES),
    ]
    """The index or sequence of indices to get items from the sequence."""


class SequenceGetItemOutputRefs(OutputRefs):
    """Output references for the GetItem operation."""

    gathered: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: (
                Sequence(
                    feature=get_sequence_feature(i.sequence.feature_),
                    length=get_sequence_length(i.index.feature_),
                )
                if check_feature_is_sequence(i.index.feature_)
                else get_sequence_feature(i.sequence.feature_)
            )
        ),
    ]
    """The feature reference to the gathered items from the sequence."""


class SequenceGetItemConfig(BaseDataProcessorConfig):
    """Configuration class for the GetItem operation."""

    ...


class SequenceGetItem(
    BaseDataProcessor[
        SequenceGetItemConfig,
        SequenceGetItemInputRefs,
        SequenceGetItemOutputRefs,
    ]
):
    """The Sequence GetItem Data Processor.

    This class defines the operation to get items from a sequence feature based
    on the given index or indices.
    """

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Process a batch of data for the GetItem operation.

        Args:
            inputs (Batch): The input batch containing features 'a' and 'b'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the binary operation.
        """
        op = (
            (lambda s, idx: list(map(s.__getitem__, idx)))
            if check_feature_is_sequence(io.inputs["index"])
            else (lambda s, i: s[i])
        )

        return {
            "gathered": list(
                starmap(op, zip(inputs["sequence"], inputs["index"]))
            )
        }


class SequenceSetItemInputRefs(InputRefs):
    """Input references for the SetItem operation."""

    sequence: Annotated[FeatureRef, CheckFeatureEquals(Sequence)]
    """The sequence feature reference."""

    # either an index or a sequence of indices
    index: Annotated[
        FeatureRef,
        CheckFeatureEquals(INDEX_TYPES) | CheckFeatureIsSequence(INDEX_TYPES),
    ]
    """The index or sequence of indices to set items in the sequence."""

    value: Annotated[FeatureRef, AnyFeatureType()]
    """The value or sequence of values to set at the specified indices in the sequence."""

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization check to ensure values align for setting in the sequence.

        Args:
            __context: The context for the model initialization.

        Raises:
            TypeError: If the values do not match the required type or length.
        """
        if check_feature_is_sequence(self.index.feature_):
            # make sure the values match the other inputs, i.e.
            #  - values must be a sequence
            #  - values sequence must have the same length as the index
            #  - values must be of the same type as the values in the sequence
            if (
                (not check_feature_is_sequence(self.value.feature_))
                or (
                    (get_sequence_length(self.value.feature_) != -1)
                    and (get_sequence_length(self.index.feature_) != -1)
                    and (
                        get_sequence_length(self.value.feature_)
                        != get_sequence_length(self.index.feature_)
                    )
                )
                or (
                    not check_feature_equals(
                        get_sequence_feature(self.value.feature_),
                        get_sequence_feature(self.sequence.feature_),
                    )
                )
            ):
                raise TypeError(
                    "Values must match the sequence type and length."
                )

        else:
            # values must be a single value of the correct type
            if not check_feature_equals(
                self.value.feature_,
                get_sequence_feature(self.sequence.feature_),
            ):
                raise TypeError("Value must match the sequence type.")


class SequenceSetItemOutputRefs(OutputRefs):
    """Output references for the SetItem operation."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(lambda _, i: i.sequence.feature_)
    ]
    """The feature reference to the result of the SetItem operation."""


class SequenceSetItemConfig(BaseDataProcessorConfig):
    """Configuration class for the SetItem operation."""

    ignore_length_mismatch: bool = False
    """Whether to ignore length mismatch between index and value sequences."""


class SequenceSetItem(
    BaseDataProcessor[
        SequenceSetItemConfig,
        SequenceSetItemInputRefs,
        SequenceSetItemOutputRefs,
    ]
):
    """The Sequence SetItem Data Processor.

    This class defines the operation to set items in a sequence feature based
    on the given index or indices.
    """

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Process a batch of data for the SetItem operation.

        Args:
            inputs (Batch): The input batch containing features 'a' and 'b'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the binary operation.
        """
        op = (
            (lambda s, idx: list(map(s.__getitem__, idx)))
            if check_feature_is_sequence(io.inputs["index"])
            else (lambda s, i: s[i])
        )

        if (
            not self.config.ignore_length_mismatch
            and check_feature_is_sequence(io.inputs["index"])
            and (
                (get_sequence_length(io.inputs["index"]) == -1)
                or (get_sequence_length(io.inputs["value"]) == -1)
            )
        ):
            # make sure the length of the value and index list match
            if any(
                len(vals) != len(idx)
                for vals, idx in zip(inputs["value"], inputs["index"])
            ):
                raise RuntimeError()  # TODO: write error message

        # convert sequences to numpy arrays and create all put operations
        sequences = list(
            map(partial(np.asarray, dtype=object), inputs["sequence"])
        )
        operations = starmap(
            np.put, zip(sequences, inputs["index"], inputs["value"])
        )
        # efficiently exhaust operations iterator, basically run all operations
        deque(operations, maxlen=0)

        # convert arrays with new values back to python lists
        return {"result": list(map(np.ndarray.tolist, sequences))}
