"""Module implementing the MultiLabelConfusionMatrix aggregator for multilabel classification tasks.

This module contains classes and functions to compute and manage the confusion matrix for
multilabel classification using scikit-learn's `multilabel_confusion_matrix`. It integrates
with the HypED data flow framework, allowing the computation of the confusion matrix over
a dataset in a distributed and scalable manner.

Classes:
    - MultiLabelConfusionMatrixConfig: Configuration class for
        MultiLabelConfusionMatrix aggregator.
    - MultiLabelConfusionMatrixInputRefs: Defines the input
        references required for the aggregator.
    - MultiLabelConfusionMatrixOutputRefs: Defines the output
        references produced by the aggregator.
    - MultiLabelConfusionMatrix: Implements the core functionality
        for computing and aggregating multilabel confusion matrices.

Functions:
    - infer_confusion_matrix_output_feature: Infers the output feature type for
        the aggregator based on configuration and inputs.

Usage:
    The `MultiLabelConfusionMatrix` aggregator computes the multilabel confusion matrix
        for the input ground truth and predicted labels, and aggregates the results over the
            dataset. It can be configured to handle specific classes using the `labels`
            parameter, and it outputs a 3D array representing the confusion matrix for each class.
"""
from typing import Annotated

import numpy as np
from datasets import Array3D, Value
from sklearn.metrics import multilabel_confusion_matrix
from typing_extensions import Unpack

from hyped.common.feature_checks import get_sequence_length
from hyped.data.flow.core.nodes.aggregator import (
    BaseDataAggregator,
    BaseDataAggregatorConfig,
    Batch,
    IOContext,
)
from hyped.data.flow.core.refs.inputs import CheckFeatureIsSequence, InputRefs
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class MultiLabelConfusionMatrixConfig(BaseDataAggregatorConfig):
    """Configuration for the MultiLabelConfusionMatrix Aggregator."""

    labels: list | None = None
    """A list of classes or column indices to select some
    (or to force inclusion of classes absent from the data).

    List of length n_classes.
    """


class MultiLabelConfusionMatrixInputRefs(InputRefs):
    """Input Ref description for the MultiLabelConfusionMatrix Aggregator."""

    y_true: Annotated[FeatureRef, CheckFeatureIsSequence(Value("bool"))]
    """Ground truth (correct) target values.

    Sequence of length n_classes.
    """

    y_pred: Annotated[FeatureRef, CheckFeatureIsSequence(Value("bool"))]
    """Estimated targets as returned by a classifier.

    Sequence of length n_classes.
    """

    # TODO: sample_weight


def infer_confusion_matrix_output_feature(
    config: MultiLabelConfusionMatrixConfig,
    inputs: MultiLabelConfusionMatrixInputRefs,
):
    """Infer the output feature type for the MultiLabelConfusionMatrix processor.

    Checks the configuration if `labels` is specified, otherwise uses the input
    sequence length to determine the number of classes in the confusion matrix.
    """
    if config.labels is not None:
        n_classes = len(config.labels)
    else:
        n_classes = get_sequence_length(inputs["y_pred"].feature_)

    return Array3D(shape=(n_classes, 2, 2), dtype="int64")


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
        return {
            "confusion_matrix": np.zeros(
                shape=io.outputs["confusion_matrix"].shape
            )
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
        return {"confusion_matrix": val["confusion_matrix"] + ctx}, None

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
