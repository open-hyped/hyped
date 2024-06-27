"""Utils Module for Working with Spans.

This module provides utility functions for handling and validating.
"""
import numpy as np
from datasets.features.features import FeatureType

from hyped.common.feature_checks import (
    INDEX_TYPES,
    get_sequence_feature,
    raise_feature_is_sequence,
)
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


def validate_spans_feature(ref: FeatureRef, feature: FeatureType) -> None:
    """Validate that the feature is a sequence of spans.

    Args:
        ref (FeatureRef): Reference to the feature.
        feature (FeatureType): The feature type to validate.

    Raises:
        TypeError: If the feature is not a valid sequence of spans.
    """
    raise_feature_is_sequence(ref, feature)
    raise_feature_is_sequence(ref, get_sequence_feature(feature), INDEX_TYPES)
