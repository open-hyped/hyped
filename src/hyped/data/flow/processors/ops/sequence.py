"""Module containing processor implementations for sequence operators."""
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections import deque
from functools import partial
from itertools import starmap
from typing import Any, ClassVar, TypeVar, Unpack

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
    GlobalValidator,
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import (
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.core.refs.ref import FeatureRef

from .binary import BaseBinaryOp, BaseBinaryOpConfig, BaseBinaryOpOutputRefs
from .unary import (
    BaseUnaryOp,
    BaseUnaryOpConfig,
    BaseUnaryOpOutputRefs,
    UnaryOpInputRefs,
)


class SequenceLengthInputRefs(UnaryOpInputRefs):
    """Input references for the Sequence Length operation."""

    a: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The sequence feature reference to get the length of."""


class SequenceLengthOutputRefs(BaseUnaryOpOutputRefs):
    """Output references for the Sequence Length operation."""

    result: Annotated[FeatureRef, OutputFeature(Value("int64"))]
    """The feature reference to the length of the sequence."""


class SequenceLengthConfig(BaseUnaryOpConfig):
    """Configuration class for the Sequence Length operation."""


class SequenceLength(
    BaseUnaryOp[
        SequenceLengthConfig, SequenceLengthInputRefs, SequenceLengthOutputRefs
    ]
):
    """Sequence Length Data Processor.

    This class defines the operation to get the length of a sequence feature.
    """

    op = len


class SequenceConcatConfig(BaseBinaryOpConfig):
    """Configuration class for the Concat operation."""


def validate_concat_input_refs(
    config: SequenceConcatConfig, refs: SequenceConcatInputRefs
) -> None:
    """Ensure sequence features align for concatenation.

    Args:
        config (SequenceConcatConfig): Config instance.
        refs (SequenceConcatInputRefs): Input references to validate.

    Raises:
        RuntimeError: If the sequence features do not match.
    """
    # sequence features must align in
    # order to concatenate sequences
    if not check_feature_equals(
        get_sequence_feature(refs["a"].feature_),
        get_sequence_feature(refs["b"].feature_),
    ):
        raise RuntimeError(
            f"The sequence features of 'a' and 'b' must match to perform concatenation, "
            f"got `{get_sequence_feature(refs['a'].feature_)}` "
            f"!= `{get_sequence_feature(refs['b'].feature_)}`"
        )


class SequenceConcatInputRefs(
    Annotated[InputRefs, GlobalValidator(validate_concat_input_refs)]
):
    """Input references for the Concat operation."""

    a: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The first sequence feature reference."""

    b: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The second sequence feature reference."""


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
    a_length = get_sequence_length(inputs["a"].feature_)
    b_length = get_sequence_length(inputs["b"].feature_)
    # compute the length of the output sequence
    length = -1 if -1 in (a_length, b_length) else a_length + b_length
    # build the output sequence feature assuming that
    # the input sequence features match
    return Sequence(
        feature=get_sequence_feature(inputs["a"].feature_), length=length
    )


class SequenceConcatOutputRefs(BaseBinaryOpOutputRefs):
    """Output references for the Concat operation."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(infer_concat_output_dtype)
    ]
    """The feature reference to the result of the concatenation operation."""


class SequenceConcat(
    BaseBinaryOp[
        SequenceConcatConfig, SequenceConcatInputRefs, SequenceConcatOutputRefs
    ]
):
    """Sequence Concatente Data Processor.

    This class defines the concatenation operation for sequence features.
    """

    op = operator.concat


class SequenceGetItemInputRefs(InputRefs):
    """Input references for the GetItem operation."""

    sequence: Annotated[FeatureRef, CheckFeatureIsSequence()]
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
                    feature=get_sequence_feature(i["sequence"].feature_),
                    length=get_sequence_length(i["index"].feature_),
                )
                if check_feature_is_sequence(i["index"].feature_)
                else get_sequence_feature(i["sequence"].feature_)
            )
        ),
    ]
    """The feature reference to the gathered items from the sequence."""


class SequenceGetItemConfig(BaseDataProcessorConfig):
    """Configuration class for the GetItem operation."""


class SequenceGetItem(
    BaseDataProcessor[
        SequenceGetItemConfig,
        SequenceGetItemInputRefs,
        SequenceGetItemOutputRefs,
    ]
):
    """Sequence GetItem Data Processor.

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

    def call(
        self, **kwargs: Unpack[SequenceGetItemInputRefs]
    ) -> SequenceGetItemOutputRefs:
        """Add the GetItem node to the data flow.

        This method processes the input references for the GetItem operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            sequence (FeatureRef): The input sequence from which to extract an item.
            index (FeatureRef): The index at which to extract the item from the sequence.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            SequenceGetItemOutputRefs: The output references produced by the GetItem data processor.
        """
        return super(SequenceGetItem, self).call(**kwargs)


def validate_setitem_input_refs(
    config: SequenceSetItemConfig, refs: SequenceSetItemInputRefs
) -> None:
    """Ensure values align for setting in the sequence.

    Args:
        config (SequenceSetItemConfig): The configuration for the Concat operation.
        refs (SequenceSetItemInputRefs): The input references for the Concat operation.

    Raises:
        TypeError: If the values do not match the required type or length.
    """
    if check_feature_is_sequence(refs["index"].feature_):
        # make sure the values match the other inputs, i.e.
        #  - values must be a sequence
        #  - values sequence must have the same length as the index
        #  - values must be of the same type as the values in the sequence
        # TODO: support broadcasting of values, i.e. values can be a single value
        #       even if the index is a sequence of indices, np.put implements the
        #       broadcasting logic anyways
        if (
            (not check_feature_is_sequence(refs["value"].feature_))
            or (
                (get_sequence_length(refs["value"].feature_) != -1)
                and (get_sequence_length(refs["index"].feature_) != -1)
                and (
                    get_sequence_length(refs["value"].feature_)
                    != get_sequence_length(refs["index"].feature_)
                )
            )
            or (
                not check_feature_equals(
                    get_sequence_feature(refs["value"].feature_),
                    get_sequence_feature(refs["sequence"].feature_),
                )
            )
        ):
            raise TypeError("Values must match the sequence type and length.")

    else:
        # values must be a single value of the correct type
        if not check_feature_equals(
            refs["value"].feature_,
            get_sequence_feature(refs["sequence"].feature_),
        ):
            raise TypeError("Value must match the sequence type.")


class SequenceSetItemInputRefs(
    Annotated[InputRefs, GlobalValidator(validate_setitem_input_refs)]
):
    """Input references for the SetItem operation."""

    sequence: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The sequence feature reference."""

    # either an index or a sequence of indices
    index: Annotated[
        FeatureRef,
        CheckFeatureEquals(INDEX_TYPES) | CheckFeatureIsSequence(INDEX_TYPES),
    ]
    """The index or sequence of indices to set items in the sequence."""

    value: Annotated[FeatureRef, AnyFeatureType()]
    """The value or sequence of values to set at the specified indices in the sequence."""


class SequenceSetItemOutputRefs(OutputRefs):
    """Output references for the SetItem operation."""

    result: Annotated[
        FeatureRef, LambdaOutputFeature(lambda _, i: i["sequence"].feature_)
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
    """Sequence SetItem Data Processor.

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

    def call(
        self, **kwargs: Unpack[SequenceSetItemInputRefs]
    ) -> SequenceSetItemOutputRefs:
        """Add the SetItem node to the data flow.

        This method processes the input references for the SetItem operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            sequence (FeatureRef): The input sequence in which items will be set.
            index (FeatureRef): The index or indices at which to set the items in the sequence.
            value (FeatureRef): The value or values to set at the specified indices in the sequence.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            SequenceSetItemOutputRefs: The output references produced by the SetItem data processor.
        """
        return super(SequenceSetItem, self).call(**kwargs)


def validate_seqvalop_input_refs(
    config: BaseSequenceValueOpConfig, refs: SequenceValueOpInputRefs
) -> None:
    """Post-initialization check to ensure value matches the feature type of the sequence values.

    Args:
        config (BaseSequenceValueOpConfig): The configuration for the Concat operation.
        refs (SequenceValueOpInputRefs): The input references for the Concat operation.

    Raises:
        TypeError: If the value feature does not match the feature type of the sequence values.
    """
    # make sure the value matches the feature type of the sequence values
    if not check_feature_is_sequence(
        refs["sequence"].feature_, refs["value"].feature_
    ):
        raise TypeError(
            "Value feature type does not match the sequence feature type."
        )


class SequenceValueOpInputRefs(
    Annotated[InputRefs, GlobalValidator(validate_seqvalop_input_refs)]
):
    """Input references for sequence value operations."""

    sequence: Annotated[FeatureRef, CheckFeatureIsSequence()]
    """The reference to the sequence feature."""

    value: Annotated[FeatureRef, AnyFeatureType()]
    """The reference to the value feature to be checked against the sequence values."""


class BaseSequenceValueOpConfig(BaseDataProcessorConfig):
    """Base Configuration class for sequence-value operations."""


C = TypeVar("C", bound=BaseSequenceValueOpConfig)
I = TypeVar("I", bound=SequenceValueOpInputRefs)
O_ = TypeVar("O_", bound=OutputRefs)


class BaseSequenceValueOp(BaseDataProcessor[C, I, O_], ABC):
    """Base class for sequence value operations.

    This class provides a template for processing operations that involve a sequence and a value.

    Attributes:
        _OUTPUT_KEY (str): The key for the output feature in the result batch.
    """

    _OUTPUT_KEY: ClassVar[str]

    @abstractmethod
    def op(self, seq: list[Any], val: Any) -> Any:
        """The sequence-value operation to apply."""

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Processes a batch of inputs, applying the sequence-value operation.

        Args:
            inputs (Batch): The input batch containing features 'a' and 'b'.
            index (list[int]): The indices of the batch.
            rank (int): The rank of the current process.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Batch: The batch containing the result of the sequence-value operation.
        """
        return {
            type(self)._OUTPUT_KEY: [
                self.op(a, b)
                for a, b in zip(inputs["sequence"], inputs["value"])
            ]
        }

    def call(self, **kwargs: Unpack[SequenceValueOpInputRefs]) -> O_:
        """Add the sequence-value operation node to the data flow.

        This method processes the input references for the sequence-value operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            sequence (FeatureRef): The reference to the sequence feature.
            value (FeatureRef): The reference to the value feature to be checked against the sequence values.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            O: The output references produced by the sequence-value data processor.
        """
        return super(BaseSequenceValueOp, self).call(**kwargs)


class SequenceContainsOutputRefs(OutputRefs):
    """Output references for the :code:`contains` operation."""

    contains: Annotated[FeatureRef, OutputFeature(Value("bool"))]
    """The feature reference to the result of the :code:`contains` operation."""


class SequenceContainsConfig(BaseSequenceValueOpConfig):
    """Configuration class for the :code:`contains` operation."""

    """The :code:`contains` operation."""


class SequenceContains(
    BaseSequenceValueOp[
        SequenceContainsConfig,
        SequenceValueOpInputRefs,
        SequenceContainsOutputRefs,
    ]
):
    """Sequence Contains Data Drocessor.

    This class defines the operation that checks if a value is contained within a sequence.
    """

    _OUTPUT_KEY: ClassVar[str] = "contains"
    op = operator.contains


class SequenceCountOfOutputRefs(OutputRefs):
    """Output references for the :code:`countOf` operation."""

    count: Annotated[FeatureRef, OutputFeature(Value("int64"))]
    """The feature reference to the result of the :code:`countOf` operation."""


class SequenceCountOfConfig(BaseSequenceValueOpConfig):
    """Configuration class for the :code:`countOf operation."""


class SequenceCountOf(
    BaseSequenceValueOp[
        SequenceCountOfConfig,
        SequenceValueOpInputRefs,
        SequenceCountOfOutputRefs,
    ]
):
    """:code:`countOf` Data Processor.

    This class defines the operation that counts occurrences of a value within a sequence.
    """

    _OUTPUT_KEY: ClassVar[str] = "count"
    op = operator.countOf


class SequenceIndexOfOutputRefs(OutputRefs):
    """Output references for the :code:`indexOf` operation."""

    index: Annotated[FeatureRef, OutputFeature(Value("int64"))]
    """The feature reference to the result of the :code:`indexOf` operation."""


class SequenceIndexOfConfig(BaseSequenceValueOpConfig):
    """Configuration class for the :code:`indexOf` operation."""


class SequenceIndexOf(
    BaseSequenceValueOp[
        SequenceIndexOfConfig,
        SequenceValueOpInputRefs,
        SequenceIndexOfOutputRefs,
    ]
):
    """:code:`indexOf` Data Processor.

    This class defines the operation that finds the index of a value within a sequence.
    """

    _OUTPUT_KEY: ClassVar[str] = "index"
    op = operator.indexOf


class MultiSequenceOpInputRefs(InputRefs):
    """Input references for MultiSequenceOp."""

    sequences: Annotated[FeatureRef, CheckFeatureIsSequence(Sequence)]
    """The sequence of input sequences to process. This is validated to be a nested sequence."""


class BaseMultiSequenceOpOutputRefs(OutputRefs):
    """Output references for MultiSequenceOp."""

    result: Annotated[FeatureRef, OutputFeature(None)]
    """A reference to the result output feature."""


class BaseMultiSequenceOpConfig(BaseDataProcessorConfig):
    """Configuration for MultiSequenceOp."""


C = TypeVar("C", bound=BaseMultiSequenceOpConfig)
I = TypeVar("I", bound=MultiSequenceOpInputRefs)
O = TypeVar("O", bound=BaseMultiSequenceOpOutputRefs)


class BaseMultiSequenceOp(BaseDataProcessor[C, I, O], ABC):
    """Base class for multi-sequence operations.

    Inherits from BaseDataProcessor to process batches of sequences using a specified operation.
    """

    @abstractmethod
    def op(self, *args: list[Any]) -> list[Any]:
        """The operation to be performed on the sequences."""

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Process a batch of input sequences using the configured operation.

        Args:
            inputs (Batch): The input batch containing sequences.
            index (list[int]): List of indices for the current batch.
            rank (int): Rank of the process.
            io (IOContext): IO context for processing.

        Returns:
            Batch: The processed batch with the result of the operation.
        """
        return {
            "result": [list(self.op(*seqs)) for seqs in inputs["sequences"]]
        }

    def call(
        self, **kwargs: Unpack[MultiSequenceOpInputRefs]
    ) -> BaseMultiSequenceOpOutputRefs:
        """Add the multi-sequence operation node to the data flow.

        This method processes the input references for the multi-sequence operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            sequences (FeatureRef): The input sequences to process. This is validated to be a nested sequence.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            BaseMultiSequenceOpOutputRefs: The output references produced by the multi-sequence data processor.
        """
        return super(BaseMultiSequenceOp, self).call(**kwargs)


class SequenceZipOutputRefs(BaseMultiSequenceOpOutputRefs):
    """Output references for SequenceZip operation."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                Sequence(
                    get_sequence_feature(i["sequences"].feature_).feature,
                    length=get_sequence_length(i["sequences"].feature_),
                ),
                length=get_sequence_length(
                    get_sequence_feature(i["sequences"].feature_)
                ),
            )
        ),
    ]
    """A reference to the zipped sequence."""


class SequenceZipConfig(BaseMultiSequenceOpConfig):
    """Configuration for SequenceZip operation."""


class SequenceZip(
    BaseMultiSequenceOp[
        SequenceZipConfig, MultiSequenceOpInputRefs, SequenceZipOutputRefs
    ]
):
    """Data Processor for zipping sequences."""

    op = zip
