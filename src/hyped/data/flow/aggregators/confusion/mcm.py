"""Module implementing the :class:`MultiLabelConfusionMatrix` aggregator.

This module contains classes and functions to compute the matrix in a multilabel
classification setting using :code:`scikit-learn`'s :code:`multilabel_confusion_matrix`.
"""
from __future__ import annotations

from typing import Annotated

import numpy as np
from datasets import ClassLabel, Sequence, Value
from sklearn.metrics import multilabel_confusion_matrix
from typing_extensions import Unpack

from hyped.common.feature_checks import (
    check_feature_equals,
    check_feature_is_sequence,
    check_sequence_lengths_match,
    get_sequence_length,
    get_sequence_shape,
)
from hyped.data.flow.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    GlobalValidator,
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class MultiLabelConfusionMatrixConfig(BaseDataAggregatorConfig):
    """Configuration for the MultiLabelConfusionMatrix Aggregator."""

    labels: list[int] | None = None
    """A list of classes or column indices to select some
    (or to force inclusion of classes absent from the data).

    List of length :code:`n_classes`.
    """


def validate_input_sequences(
    config: MultiLabelConfusionMatrixConfig,
    input_refs: MultiLabelConfusionMatrixInputRefs,
) -> None:
    """Validates that the input sequences for the :class:`MultiLabelConfusionMatrix` aggregator are compatible.

    This function checks the compatibility of the :code:`y_true` and :code:`y_pred` input features to ensure
    they are appropriate for multi-label classification.

    Args:
        config (MultiLabelConfusionMatrixConfig): Configuration object for the MultiLabelConfusionMatrix.
        input_refs (InputRefs): A reference to the input features (y_true, y_pred) to validate.

    Raises:
        RuntimeError: If the input features do not meet the required compatibility criteria.
    """
    y_true = input_refs["y_true"].feature_
    y_pred = input_refs["y_pred"].feature_
    # check which features are sequences
    y_true_is_sequence = check_feature_is_sequence(y_true)
    y_pred_is_sequence = check_feature_is_sequence(y_pred)

    if y_true_is_sequence and y_pred_is_sequence:
        assert get_sequence_length(y_true) != -1
        assert get_sequence_length(y_pred) != -1
        # TODO: use allow_arbitrary_lengths arg of `check_sequence_lengths_match` instead of asserts
        if not check_sequence_lengths_match(y_true, y_pred):
            raise RuntimeError(
                "Sequence length of y_true must match sequence length of y_pred "
                f"Got y_true with len={get_sequence_length(y_true)} "
                f"and y_pred with len={get_sequence_length(y_pred)}"
            )

    elif y_true_is_sequence ^ y_pred_is_sequence:
        raise RuntimeError(
            "'MultiLabelConfusionMatrix' doesn't support a mix of "
            "multiclass- and multilabel-indicator targets."
        )

    elif not check_feature_equals(y_true, y_pred):
        raise RuntimeError(
            "ClassLabel features of y_true and y_pred must match. "
            f"Got y_true with {input_refs['y_true'].feature_} "
            f"and y_pred with {input_refs['y_pred'].feature_}"
        )


class MultiLabelConfusionMatrixInputRefs(
    Annotated[InputRefs, GlobalValidator(validate_input_sequences)]
):
    """Input Ref description for the MultiLabelConfusionMatrix Aggregator."""

    y_true: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool")) | CheckFeatureEquals(ClassLabel),
    ]
    """Ground truth (correct) target values.

    Either label indicator or a sequence of shape (n_classes,)
    """

    y_pred: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(Value("bool")) | CheckFeatureEquals(ClassLabel),
    ]
    """Estimated targets as returned by a classifier.

    Either label indicator or a sequence of shape (n_classes,)
    """


def infer_confusion_matrix_output_feature(
    config: MultiLabelConfusionMatrixConfig,
    inputs: MultiLabelConfusionMatrixInputRefs,
) -> Sequence:
    """Infer the output feature type for the MultiLabelConfusionMatrix processor.

    Checks the configuration if `labels` is specified, otherwise uses the input
    sequence length to determine the number of classes in the confusion matrix.
    """
    if config.labels is not None:
        n_classes = len(config.labels)
    elif check_feature_is_sequence(inputs["y_pred"].feature_):
        n_classes = get_sequence_length(inputs["y_pred"].feature_)
    else:
        n_classes = len(inputs["y_pred"].feature_.names)

    return Sequence(
        Sequence(
            Sequence(Value("int64"), length=2),
            length=2,
        ),
        length=n_classes,
    )


class MultiLabelConfusionMatrixOutputRefs(OutputRefs):
    """Output Ref description for the MultiLabelConfusionMatrix Aggregator."""

    confusion_matrix: Annotated[
        FeatureRef, LambdaOutputFeature(infer_confusion_matrix_output_feature)
    ]


class MultiLabelConfusionMatrix(
    BaseDataAggregator[
        MultiLabelConfusionMatrixConfig,
        MultiLabelConfusionMatrixInputRefs,
        MultiLabelConfusionMatrixOutputRefs,
    ]
):
    """The MultiLabelConfusionMatrix Aggregator.

    Implements sklearn's `multilabel_confusion_matrix` as a hyped aggregator.
    The Output is a 3DArray, which corresponds to the aggregated output of
    `multilabel_confusion_matrix` over the whole dataset.
    """

    def initialize(self, io: IOContext) -> tuple[dict[str, float], None]:
        """Initialize the confusion matrix to zeros."""
        shape = get_sequence_shape(io.outputs["confusion_matrix"])
        return {
            "confusion_matrix": np.zeros(shape=shape, dtype=np.int64).tolist()
        }, None

    async def extract(
        self, inputs: Batch, index: list[int], rank: int, io: IOContext
    ) -> np.ndarray:
        """Computes the confusion matrix for the current batch."""
        return multilabel_confusion_matrix(
            y_true=inputs["y_true"],
            y_pred=inputs["y_pred"],
            labels=self.config.labels,
        )

    async def update(
        self, val: float, ctx: np.ndarray, state: None, io: IOContext
    ) -> tuple[dict[str, float], None]:
        """Updates the confusion matrix by addition."""
        return {
            "confusion_matrix": (
                np.array(val["confusion_matrix"]) + ctx
            ).tolist()
        }, None

    def call(
        self, **kwargs: Unpack[MultiLabelConfusionMatrixInputRefs]
    ) -> MultiLabelConfusionMatrixOutputRefs:
        """Adds the processor to the data flow.

        This method first prepares the inputs, then adds the processor to the data
        flow and returns a feature reference to the output features of the processor.

        Args:
            y_true (FeatureRef): Ground truth (correct) target values.
            y_pred (FeatureRef): Estimated targets as returned by a classifier.
            **kwargs (FeatureRef): Keyword arguments specifying feature references to be passed
                as inputs to the processor.

        Returns:
            MultiLabelConfusionMatrixOutputRefs: The output references produced by the processor.
        """
        return super().call(**kwargs)
