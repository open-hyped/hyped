"""Data Processor for converting character spans to token spans.

This module defines the functionality required to process character spans and convert them into
token spans, which are useful for various Natural Language Processing (NLP) tasks such as Named
Entity Recognition (NER).
"""
import numpy as np
from datasets.features.features import Sequence, Value
from typing_extensions import Annotated, NotRequired, Unpack

from hyped.common.typing import Index, Rank, Sample
from hyped.core.nodes.processor import BaseDataProcessor, BaseDataProcessorConfig, IOContext
from hyped.core.refs.inputs import CheckFeatureEquals, FeatureValidator, InputRefs
from hyped.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.core.refs.ref import FeatureRef

from .utils import compute_spans_overlap_matrix, validate_spans_feature


class ChrToTokSpansInputRefs(InputRefs):
    """Input references for ChrToTokSpans."""

    chr_spans: Annotated[FeatureRef, FeatureValidator(validate_spans_feature)]
    """Character spans feature reference."""

    query_spans: Annotated[FeatureRef, FeatureValidator(validate_spans_feature)]
    """Query spans feature reference."""

    special_tokens_mask: NotRequired[
        Annotated[FeatureRef, CheckFeatureEquals(Sequence(Value("int32")))]
    ]
    """Mask indicating tokens not to be mapped to queries."""


class ChrToTokSpansOutputRefs(OutputRefs):
    """Output references for ChrToTokSpans."""

    tok_spans: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                Sequence(Value("int32"), length=2),
                length=i["query_spans"].feature_.length,
            )
        ),
    ]
    """Token spans feature reference."""


class ChrToTokSpansConfig(BaseDataProcessorConfig):
    """Configuration for ChrToTokSpans."""

    include_partial_start: bool = True
    """Whether to include tokens that are only partly covered at the beginning of the query span."""

    include_partial_end: bool = True
    """Whether to include tokens that are only partly covered at the end of the query span."""


class ChrToTokSpans(
    BaseDataProcessor[ChrToTokSpansConfig, ChrToTokSpansInputRefs, ChrToTokSpansOutputRefs]
):
    """Processor to convert character spans to token spans.

    This processor computes the span overlap matrix between query spans and character spans,
    and then converts the overlapping spans into token spans.
    """

    def process(self, inputs: Sample, index: Index, rank: Rank, io: IOContext) -> Sample:
        """Process input samples to compute token spans.

        Args:
            inputs (Sample): The input sample containing character and query spans.
            index (Index): The index of the sample.
            rank (Rank): The rank of the process.
            io (IOContext): The input/output context.

        Returns:
            Sample: The output sample containing the computed token spans.
        """
        query_spans = np.asarray(inputs["query_spans"])
        chr_spans = np.asarray(inputs["chr_spans"])
        mask = inputs.get("special_tokens_mask", None)
        # compute the span overlap matrix between
        # the query spans and the character spans
        overlap = compute_spans_overlap_matrix(
            source_spans=query_spans, target_spans=chr_spans, special_tokens=mask
        )
        # get begins and ends from mask
        tok_spans_begin = overlap.argmax(axis=1)
        tok_spans_end = tok_spans_begin + overlap.sum(axis=1)

        # exclude partially overlapping tokens at the beginning
        if not self.config.include_partial_start:
            partial_mask = chr_spans[tok_spans_begin, 0] != query_spans[:, 0]
            tok_spans_begin[partial_mask] += 1

        # exclude partially overlapping tokens at the end
        if not self.config.include_partial_end:
            partial_mask = chr_spans[tok_spans_end - 1, 1] != query_spans[:, 1]
            tok_spans_end[partial_mask] -= 1

        # build output
        tok_spans = list(zip(tok_spans_begin, tok_spans_end))
        return Sample(tok_spans=tok_spans)

    def call(self, **kwargs: Unpack[ChrToTokSpansInputRefs]) -> ChrToTokSpansOutputRefs:
        """Execute the ChrToTokSpans processor.

        Processes the input references to convert character spans ('chr_spans')
        to token spans ('tok_spans') based on the overlap between query spans ('query_spans')
        and character spans.

        Args:
            chr_spans (FeatureRef): The feature reference to the sequence of character spans.
            query_spans (FeatureRef): The sequence of query spans to convert.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            ChrToTokSpansOutputRefs: The output references containing the computed token spans.
        """
        return super(ChrToTokSpans, self).call(**kwargs)
