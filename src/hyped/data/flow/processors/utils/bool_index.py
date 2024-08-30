"""Data Processor for indexing sequences using a boolean mask.

This module defines the functionality required to filter a sequence based on a boolean mask.
"""
from itertools import compress
from typing import Annotated

from datasets import Sequence, Value
from datasets.features.features import FeatureType

from hyped.common.feature_checks import (
    check_feature_is_sequence,
    get_sequence_feature,
)
from hyped.data.flow.core.nodes.base import IOContext
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.flow.core.refs.inputs import CheckFeatureIsSequence, InputRefs
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class BooleanIndexingConfig(BaseDataProcessorConfig):
    """Configuration to the :code:`BooleanIndexing` processor."""

    # No additional configuration needed


class BooleanIndexingInputRefs(InputRefs):
    """Input references to the :code:`BooleanIndexing` processor."""

    values: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(),
    ]
    """The feature reference to the sequence of values to be filtered."""

    mask: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool")),
    ]
    """The feature reference to the sequence of boolean mask values."""


def get_output_feature(
    config: BooleanIndexingConfig, inputs: BooleanIndexingInputRefs
) -> FeatureType:
    """Generate the output feature for the indexed sequence.

    Based on the input features, this function constructs the appropriate sequence feature
    with an undefined length.

    Args:
        config (BooleanIndexingConfig): The configuration (not used here).
        inputs (BooleanIndexingInputRefs): The input references containing the values and mask.

    Returns:
        FeatureType: The feature type representing the indexed sequence.
    """
    # Get the input feature to determine the value type
    input_feature = inputs["values"].feature_
    assert check_feature_is_sequence(
        input_feature
    ), "Input values must be a sequence."

    # Create the output sequence feature with the same value type but undefined length
    value_type = get_sequence_feature(input_feature)
    return Sequence(value_type, length=-1)


class BooleanIndexingOutputRefs(OutputRefs):
    """Output references to :code:`BooleanIndexing` processor."""

    indexed_values: Annotated[
        FeatureRef, LambdaOutputFeature(get_output_feature)
    ]
    """The feature reference to the indexed sequence of values."""


class BooleanIndexing(
    BaseDataProcessor[
        BooleanIndexingConfig,
        BooleanIndexingInputRefs,
        BooleanIndexingOutputRefs,
    ]
):
    """Processor to index a sequence of values using a boolean mask.

    This processor filters a sequence of values based on a corresponding sequence of
    boolean mask values, similar to the behavior of `itertools.compress`.
    """

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> Batch:
        """Process input batches to index sequences using the mask.

        This method processes a batch of input values, filtering them based on the
        corresponding mask sequence.

        Args:
            inputs (Batch): The input batch containing the values and mask.
            index (list[int]): The indices of the samples in the batch.
            rank (int): The rank of the process.
            io (IOContext): The input/output context.

        Returns:
            Batch: The output batch containing the indexed sequences.
        """
        # Get the values and mask features
        values = inputs["values"]
        mask = inputs["mask"]

        # Apply the mask to the values
        indexed_values = [
            list(compress(value_seq, mask_seq))
            for value_seq, mask_seq in zip(values, mask)
        ]

        # Return the indexed values as a new Batch
        return Batch(indexed_values=indexed_values)
