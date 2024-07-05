"""Module for BIO (Begin-In-Out) tagging processors.

This module defines the data processor to generate BIO tags from span annotations. BIO Tagging
is a common tagging scheme used in Natural Language Processing for tasks like Named Entity
Recognition.
"""
from __future__ import annotations

import numpy as np
from datasets import ClassLabel, Sequence, Value
from typing_extensions import Annotated, Unpack

from hyped.common.feature_checks import (
    INT_TYPES,
    UINT_TYPES,
    check_feature_equals,
    check_feature_is_sequence,
    get_sequence_feature,
    get_sequence_length,
)
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    IOContext,
    Sample,
)
from hyped.data.flow.core.refs.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    FeatureValidator,
    GlobalValidator,
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef

from .utils import validate_spans_feature


def validate_bio_tag_input_refs(refs: BioTagsInputRefs) -> None:
    """Global validator for bio tag input refs.

    Ensures that the lengths of spans and labels match if
    they are both specified and not dynamic.

    Args:
        refs (BioTagsInputRefs): input references to validate

    Raises:
        RuntimeError: If the lengths of spans and labels do not match.
    """
    if (
        (get_sequence_length(refs.spans.feature_) != -1)
        and (get_sequence_length(refs.labels.feature_) != -1)
    ) and (
        get_sequence_length(refs.spans.feature_)
        != get_sequence_length(refs.labels.feature_)
    ):
        raise RuntimeError(
            f"Mismatch in sequence lengths: 'spans' and 'labels' "
            f"must have the same length, got {get_sequence_length(refs.spans.feature_)} "
            f"!= {get_sequence_length(refs.labels.feature_)}."
        )


def check_input_lengths(config: BioTagsConfig, refs: BioTagsInputRefs) -> None:
    """Validate that the sequence lengths of spans and labels match.

    This function checks the sequence lengths of the :code:`spans` and :code:`labels` features
    in the provided :code:`BioTagsInputRefs`. If both sequence lengths are known and they do not
    match, it raises a :code:`RuntimeError`.

    Args:
        config (BioTagsConfig): The configuration for the BioTags processor.
        refs (BioTagsInputRefs): The input references containing the features to be validated.

    Raises:
        RuntimeError: If the sequence lengths of 'spans' and 'labels' do not match.
    """
    spans_length = get_sequence_length(refs["spans"].feature_)
    labels_length = get_sequence_length(refs["labels"].feature_)

    if (
        (spans_length != -1)
        and (labels_length != -1)
        and (spans_length != labels_length)
    ):
        raise RuntimeError(
            f"Span and label sequence length don't match, got {spans_length} != {labels_length}."
        )


class BioTagsInputRefs(
    Annotated[InputRefs, GlobalValidator(check_input_lengths)]
):
    """Input references for the BioTags processor."""

    spans: Annotated[FeatureRef, FeatureValidator(validate_spans_feature)]
    """The feature reference to the span annotations. Must be a sequence of spans."""

    labels: Annotated[
        FeatureRef, CheckFeatureIsSequence([Value("string"), ClassLabel])
    ]
    """The feature reference for the label annotations, which should be a sequence
    of :code:`strings` or :class:`ClassLabels`.
    """

    length: Annotated[FeatureRef, CheckFeatureEquals(INT_TYPES + UINT_TYPES)]
    """The feature reference to the length, which should be of integer type
    indicating the target length of the tags sequence.
    """


def build_bio_tags_feature(
    config: BioTagsConfig, inputs: BioTagsInputRefs
) -> Sequence:
    """Builds the BIO tags feature from the input spans and labels.

    Args:
        config (BioTagsConfig): The configuration for the BioTags processor.
        inputs (BioTagsInputRefs): The input references containing spans and labels.

    Returns:
        Sequence: The sequence feature representing the BIO tags.
    """
    # TODO: we can infer the length of the tags sequence
    #       from the length input feature in case the length
    #       feature comes from a constant

    # read labels feature
    labels_feature = get_sequence_feature(inputs["labels"].feature_)

    # keep string feature if input labels are also strings
    if check_feature_equals(labels_feature, Value("string")):
        return Sequence(Value("string"))

    assert check_feature_equals(labels_feature, ClassLabel)
    # build class labels from source class labels
    bio_tags = ClassLabel(
        names=[config.out_tag]
        + [
            "%s%s" % (prefix, label)
            for label in labels_feature.names
            for prefix in [config.begin_tag_prefix, config.in_tag_prefix]
        ]
    )

    return Sequence(bio_tags)


class BioTagsOutputRefs(OutputRefs):
    """Output references for the BioTags processor."""

    tags: Annotated[FeatureRef, LambdaOutputFeature(build_bio_tags_feature)]
    """The feature reference for the BIO tags."""


class BioTagsConfig(BaseDataProcessorConfig):
    """Configuration for the BioTags processor."""

    begin_tag_prefix: str = "B-"
    """The prefix for the begin tag. Defaults to "B-"."""
    in_tag_prefix: str = "I-"
    """The prefix for the in tag. Defaults to "I-"."""
    out_tag: str = "O"
    """The tag for outside spans. Defaults to "O"."""


class BioTags(
    BaseDataProcessor[BioTagsConfig, BioTagsInputRefs, BioTagsOutputRefs]
):
    """Processor for generating BIO tags from spans and labels.

    This processor takes spans and labels as input and generates BIO tags,
    """

    def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        """Processes the inputs to generate BIO tags.

        Args:
            inputs (Sample): The input sample containing spans and labels.
            index (int): The index of the current sample.
            rank (int): The rank of the current sample.
            io (IOContext): The IO context for handling input and output features.

        Returns:
            Sample: The output sample containing BIO tags.

        Raises:
            ValueError: If there is an overlap between entities.
        """
        spans = inputs["spans"]
        labels = inputs["labels"]
        length = inputs["length"]

        # convert labels to strings
        if check_feature_is_sequence(io.inputs["labels"], ClassLabel):
            labels = io.inputs["labels"].feature.int2str(labels)

        # build initial tag sequence of all out and invalid tags
        tags = np.full(length, fill_value=self.config.out_tag, dtype=object)

        # insert all entity spans
        for label, (b, e) in zip(labels, spans):
            # check for overlaps with previous annotations
            if (tags[b:e] != self.config.out_tag).any():
                # get the overlapping entity types
                overlap_types = [label] + [
                    (
                        tag.removeprefix(
                            self.config.begin_tag_prefix
                        ).removeprefix(self.config.in_tag_prefix)
                    )
                    for tag in tags[b:e]
                    if tag != self.config.out_tag
                ]
                # raise error on overlap
                raise ValueError(
                    "Detected overlap between entities of types %s"
                    % ", ".join(overlap_types)
                )

            # add entity to tag sequence
            tags[b:e] = "%s%s" % (self.config.in_tag_prefix, label)
            tags[b] = "%s%s" % (self.config.begin_tag_prefix, label)

        # convert tags to list
        tags = tags.tolist()
        # convert strings to class label ids
        if check_feature_is_sequence(io.outputs["tags"], ClassLabel):
            tags = io.outputs["tags"].feature.str2int(tags)

        return Sample(tags=tags)

    def call(self, **kwargs: Unpack[BioTagsInputRefs]) -> BioTagsOutputRefs:
        """Execute the BioTags processor.

        Processes the input references to generate BIO tags based on the spans and labels provided.
        This method validates the input lengths and builds the BIO tags sequence accordingly,
        ensuring that the sequence conforms to the specified configuration.

        Args:
            spans (FeatureRef): The feature reference to the span annotations. Must be a sequence of spans.
            labels (FeatureRef): The feature reference for the label annotations, which should be a sequence
                of :code:`strings` or :class:`ClassLabels`.
            length (FeatureRef): The feature reference to the length, which should be of integer type
                indicating the target length of the tags sequence.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            BioTagsOutputRefs: The output references containing the generated BIO tags sequence.

        Raises:
            RuntimeError: If there is a mismatch in sequence lengths between 'spans' and 'labels'.
        """
        return super(BioTags, self).call(**kwargs)
