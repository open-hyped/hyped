"""Data Processor for converting string labels to class labels.

This module defines the functionality required to process string labels and convert them into
class labels.
"""
from typing import Annotated

from datasets import ClassLabel, Sequence, Value
from datasets.features.features import FeatureType
from typing_extensions import Unpack

from hyped.common.feature_checks import (
    check_feature_is_sequence,
    get_sequence_feature,
    get_sequence_length,
)
from hyped.common.typing import Batch, IndexList, Rank
from hyped.core.nodes.base import IOContext
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig
from hyped.core.refs.inputs import CheckFeatureEquals, CheckFeatureIsSequence, InputRefs
from hyped.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef


class ToClassLabelConfig(BaseDataProcessorConfig):
    """Configuration to the :code:`ToClassLabel` processor."""

    class_labels: list[str]
    """A list of string labels that will be converted to class labels."""


class ToClassLabelInputRefs(InputRefs):
    """Input references to the :code:`ToClassLabel` processor."""

    label: Annotated[
        FeatureRef,
        CheckFeatureEquals(Value("string")) | CheckFeatureIsSequence(Value("string")),
    ]
    """The feature reference to the sequence or single value of string labels to convert."""


def get_output_feature(config: ToClassLabelConfig, inputs: ToClassLabelInputRefs) -> FeatureType:
    """Generate the output feature for class labels.

    Based on the configuration and input features, this function constructs the appropriate
    `ClassLabel` feature, handling both single string values and sequences of strings.

    Args:
        config (ToClassLabelConfig): The configuration containing the class labels.
        inputs (ToClassLabelInputRefs): The input references containing the labels to convert.

    Returns:
        FeatureType: The feature type representing the class labels.
    """
    # get the input feature to convert and build the class label feature
    input_feature = inputs["label"].feature_
    class_label = ClassLabel(names=config.class_labels)
    # handel sequence inputs
    if check_feature_is_sequence(input_feature):
        class_label = Sequence(class_label, length=get_sequence_length(input_feature))
    # return the final class label feature
    return class_label


class ToClassLabelOutputRefs(OutputRefs):
    """Output references to :code:`ToClassLabel` processor."""

    class_label: Annotated[FeatureRef, LambdaOutputFeature(get_output_feature)]
    """The feature reference to the converted class labels."""


class ToClassLabel(
    BaseDataProcessor[ToClassLabelConfig, ToClassLabelInputRefs, ToClassLabelOutputRefs]
):
    """Processor to convert string labels to class labels.

    This processor converts string labels (or sequences of string labels) into corresponding
    class labels based on the provided configuration.
    """

    async def batch_process(
        self, inputs: Batch, index: IndexList, rank: Rank, io: IOContext
    ) -> Batch:
        """Process input batches to convert labels to class labels.

        This method processes a batch of input labels, converting each label into its corresponding
        class label integer based on the provided class label mapping.

        Args:
            inputs (Batch): The input batch containing the labels to convert.
            index (IndexList): The indices of the samples in the batch.
            rank (Rank): The rank of the process.
            io (IOContext): The input/output context.

        Returns:
            Batch: The output batch containing the converted class labels.
        """
        # get the class label feature to convert to
        class_label = io.outputs["class_label"]
        class_label = (
            class_label
            if isinstance(class_label, ClassLabel)
            else get_sequence_feature(class_label)
        )
        assert isinstance(class_label, ClassLabel)  # TODO: write error message
        # convert each item in the input to a class label
        return Batch(class_label=list(map(class_label.str2int, inputs["label"])))

    def call(self, **kwargs: Unpack[ToClassLabelInputRefs]) -> ToClassLabelOutputRefs:
        """Execute the :class:`ToClassLabel` processor.

        Processes the input references to convert string labels (:code:`label`) to class labels
        (:code:`class_label`) based on the mapping of string labels to class labels.

        Args:
            label (FeatureRef): The feature reference to the sequence or single value of string
                labels.
            **kwargs (FeatureRef): Keyword arguments passed to the call method.

        Returns:
            ToClassLabelOutputRefs: The output references containing the converted class labels.
        """
        return super(ToClassLabel, self).call(**kwargs)
