"""Module containing processor implementations for sequence item operators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, TypeVar

from datasets import Sequence
from typing_extensions import Annotated, Unpack

from hyped.common.feature_checks import get_sequence_length  # noqa: F401
from hyped.common.feature_checks import check_feature_is_sequence, get_sequence_feature
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import FeatureValidator, InputRefs
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class BaseMultiSequenceOpConfig(BaseDataProcessorConfig):
    """Configuration for MultiSequenceOp."""


def validate_multisequence_feat(config: BaseMultiSequenceOpConfig, feat_ref: FeatureRef) -> None:
    """Validates that the provided feature is a multisequence feature.

    This function ensures that the feature is a dictionary indexed by consecutive integers,
    starting from 0, and that the values are sequences of the same type.

    Args:
        config (BaseMultiSequenceOpConfig): The config of the MultiSequenceOp Processor.
        feat_ref (FeatureType): The reference to the feature being validated.

    Raises:
        TypeError: If the feature is not a dictionary.
        TypeError: If the keys of the dictionary are not integers.
        TypeError: If the dictionary is not indexed by consecutive integers starting from 0.
        TypeError: If the values of the dictionary are not sequences of the same type.
    """
    if not isinstance(feat_ref.feature_, dict):
        raise TypeError(
            "Expected multisequence feature `%s` to be of dict type, got %s "
            % (feat_ref.key_, type(feat_ref.feature_))
        )

    # convert all multisequence keys to integer indices
    try:
        indices = [int(key) for key in feat_ref.feature_]
    except ValueError:
        raise TypeError(
            "Expected multisequence feature `%s` to have integer-like keys, got %s "
            % (feat_ref.key_, feat_ref.feature_)
        )

    # check that multisequence is indexed by consecutive integers starting at 0
    sorted_indices = list(sorted(indices))
    if not (all(k == i for i, k in enumerate(sorted_indices)) and sorted_indices[0] == 0):
        raise TypeError(
            "Expected multisequence feature `%s` to be indexed by consecutive integers, got %s "
            % (feat_ref.key_, sorted_indices)
        )

    # TODO: Allow castable types
    value_type = next(iter(feat_ref.feature_.values())).feature
    for k in feat_ref.feature_.keys():
        if not check_feature_is_sequence(feat_ref.feature_[k], value_type):
            raise TypeError(
                "Expected `%s.%s` to be a sequence of type %s"
                ", got %s " % (feat_ref.key_, k, value_type, feat_ref.feature_[k])
            )


class MultiSequenceOpInputRefs(InputRefs):
    """Input references for MultiSequenceOp."""

    sequences: Annotated[FeatureRef, FeatureValidator(validate_multisequence_feat)]
    """The sequence of input sequences to process. This is validated to be a 
    dict of sequences indexed by consecutive integers starting from 0.
    
    Example:
        .. code-block:: python
        sequences = collect(
            {
                "0": feature_ref_1,
                "1": feature_ref_2,
                "2": feature_ref_3,
            }
        )
    """


class BaseMultiSequenceOpOutputRefs(OutputRefs):
    """Output references for MultiSequenceOp."""

    result: Annotated[FeatureRef, OutputFeature(None)]
    """A reference to the result output feature."""


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
            "result": [
                list(self.op(*(seq_dict[str(i)] for i in range(len(seq_dict)))))
                for seq_dict in inputs["sequences"]
            ]
        }

    def call(self, **kwargs: Unpack[MultiSequenceOpInputRefs]) -> O:
        """Add the multisequence operation node to the data flow.

        This method processes the input references for the multisequence operation, adds
        the corresponding node to the data flow, and returns the references to the
        output features generated by the processor.

        Args:
            sequences (FeatureRef): The reference to the sequences feature.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            O: The output references produced by the multisequence data processor.
        """
        return super(BaseMultiSequenceOp, self).call(**kwargs)


class SequenceChainConfig(BaseMultiSequenceOpConfig):
    """Configuration class for the SequenceChain operation."""


def infer_chain_output_dtype(
    config: SequenceChainConfig, inputs: MultiSequenceOpInputRefs
) -> Sequence:
    """Infer the output data type for the Chain operation.

    Args:
        config (SequenceChainConfig): The configuration for the Chain operation.
        inputs (MultiSequenceOpInputRefs): The input references for the Chain operation.

    Returns:
        Sequence: The output sequence feature with inferred length.
    """
    sequence_feature = get_sequence_feature(next(iter(inputs["sequences"].feature_.values())))
    sequence_lengths = [get_sequence_length(feat) for feat in inputs["sequences"].feature_.values()]
    return Sequence(
        feature=sequence_feature,
        length=-1 if -1 in sequence_lengths else sum(sequence_lengths),
    )


class SequenceChainOutputRefs(BaseMultiSequenceOpOutputRefs):
    """Output references for the Chain operation."""

    result: Annotated[FeatureRef, LambdaOutputFeature(infer_chain_output_dtype)]
    """The feature reference to the result of the chain operation."""


class SequenceChain(
    BaseMultiSequenceOp[SequenceChainConfig, MultiSequenceOpInputRefs, SequenceChainOutputRefs]
):
    """Sequence Chain Data Processor.

    This class defines the chain operation for sequence features.
    """

    def op(self, *args: list[Any]) -> list[Any]:
        """Chain all sequences."""
        return list(chain(*args))


class SequenceZipOutputRefs(BaseMultiSequenceOpOutputRefs):
    """Output references for SequenceZip operation."""

    result: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                Sequence(
                    get_sequence_feature(next(iter(i["sequences"].feature_.values()))),
                    length=len(i["sequences"].feature_),
                ),
                length=min(map(get_sequence_length, i["sequences"].feature_.values())),
            )
        ),
    ]
    """A reference to the zipped sequence."""


class SequenceZipConfig(BaseMultiSequenceOpConfig):
    """Configuration for SequenceZip operation."""


class SequenceZip(
    BaseMultiSequenceOp[SequenceZipConfig, MultiSequenceOpInputRefs, SequenceZipOutputRefs]
):
    """Data Processor for zipping sequences."""

    op = zip
