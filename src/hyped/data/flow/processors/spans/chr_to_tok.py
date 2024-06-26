"""Data Processor for converting character spans to token spans.

This module defines the functionality required to process character spans and convert them into
token spans, which are useful for various Natural Language Processing (NLP) tasks such as Named
Entity Recognition (NER).
"""
import numpy as np
from datasets.features.features import FeatureType, Sequence, Value
from typing_extensions import Annotated

from hyped.common.feature_checks import (
    INDEX_TYPES,
    get_sequence_feature,
    raise_feature_is_sequence,
)
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    IOContext,
    Sample,
)
from hyped.data.flow.core.refs.inputs import FeatureValidator, InputRefs
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


def compute_spans_overlap_matrix(
    source_spans: list[tuple[int]],
    target_spans: list[tuple[int]],
) -> np.ndarray:
    """Compute the span overlap matrix.

    The span overlap matrix :math:`O` is a binary matrix of shape :math:`(n, m)` where
    :math:`n` is the number of source spans and :math:`m` is the number of target spans.
    The boolean value :math:`O_{ij}` indicates whether the :math:`i`-th source span
    overlaps with the :math:`j`-th target span.

    Arguments:
        source_spans (Sequence[tuple[int]]): either a sequence of source spans or a
            single source span
        target_spans (Sequence[tuple[int]]): either a sequence of target spans or a
            single target span

    Returns:
        O (np.ndarray): binary overlap matrix of shape
        :math:`(len(source_spans), len(target_spans))`
    """
    # convert spans to numpy arrays
    source_spans = np.asarray(source_spans).reshape(-1, 2)
    target_spans = np.asarray(target_spans).reshape(-1, 2)
    # compute overlap mask
    return (
        (
            # source overlaps with target begin
            (source_spans[:, 0, None] <= target_spans[None, :, 0])
            & (target_spans[None, :, 0] < source_spans[:, 1, None])
        )
        | (
            # source overlaps with target end
            (source_spans[:, 0, None] < target_spans[None, :, 1])
            & (target_spans[None, :, 1] <= source_spans[:, 1, None])
        )
        | (
            # target is contained in source
            (source_spans[:, 0, None] <= target_spans[None, :, 0])
            & (target_spans[None, :, 1] <= source_spans[:, 1, None])
        )
        | (
            # source is contained in target
            (target_spans[None, :, 0] <= source_spans[:, 0, None])
            & (source_spans[:, 1, None] <= target_spans[None, :, 1])
        )
    )


def _validate_spans_feature(ref: FeatureRef, feature: FeatureType) -> None:
    """Validate that the feature is a sequence of spans.

    Args:
        ref (FeatureRef): Reference to the feature.
        feature (FeatureType): The feature type to validate.

    Raises:
        TypeError: If the feature is not a valid sequence of spans.
    """
    raise_feature_is_sequence(ref, feature)
    raise_feature_is_sequence(ref, get_sequence_feature(feature), INDEX_TYPES)


class ChrToTokSpansInputRefs(InputRefs):
    """Input references for ChrToTokSpans."""

    chr_spans: Annotated[FeatureRef, FeatureValidator(_validate_spans_feature)]
    """Character spans feature reference."""

    query_spans: Annotated[
        FeatureRef, FeatureValidator(_validate_spans_feature)
    ]
    """Query spans feature reference."""


class ChrToTokSpansOutputRefs(OutputRefs):
    """Output references for ChrToTokSpans."""

    tok_spans: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                Sequence(Value("int32"), length=2),
                length=i.query_spans.feature_.length,
            )
        ),
    ]
    """Token spans feature reference."""


class ChrToTokSpansConfig(BaseDataProcessorConfig):
    """Configuration for ChrToTokSpans."""


class ChrToTokSpans(
    BaseDataProcessor[
        ChrToTokSpansConfig, ChrToTokSpansInputRefs, ChrToTokSpansOutputRefs
    ]
):
    """Processor to convert character spans to token spans.

    This processor computes the span overlap matrix between query spans and character spans,
    and then converts the overlapping spans into token spans.
    """

    def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        """Process input samples to compute token spans.

        Args:
            inputs (Sample): The input sample containing character and query spans.
            index (int): The index of the sample.
            rank (int): The rank of the process.
            io (IOContext): The input/output context.

        Returns:
            Sample: The output sample containing the computed token spans.
        """
        # compute the span overlap matrix between
        # the query spans and the character spans
        overlap = compute_spans_overlap_matrix(
            source_spans=inputs["query_spans"],
            target_spans=inputs["chr_spans"],
        )
        # get begins and ends from mask
        tok_spans_begin = overlap.argmax(axis=1)
        tok_spans_end = tok_spans_begin + overlap.sum(axis=1)
        # build output
        tok_spans = list(zip(tok_spans_begin, tok_spans_end))
        return Sample(tok_spans=tok_spans)
