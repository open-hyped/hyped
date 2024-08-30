"""Module implementing the :class:`PrecisionRecallFScoreSupport` processor.

This can be used for computing precision, recall, F-score and support metrics for classification tasks.

This module contains classes and functions to calculate precision, recall, F-score, and
support metrics from a given confusion matrix. It leverages scikit-learn's underlying functions
but adapts them to operate directly on confusion matrices within the hyped data flow framework.
"""
from typing import Annotated, Literal

import numpy as np
from datasets import Sequence, Value
from datasets.features.features import FeatureType
from sklearn.metrics._classification import _prf_divide
from sklearn.utils.extmath import _nanaverage
from typing_extensions import Unpack

from hyped.common.feature_checks import get_sequence_length, get_sequence_shape
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    IOContext,
    Sample,
)
from hyped.data.flow.core.refs.inputs import FeatureValidator, InputRefs
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef


class PrecisionRecallFScoreSupportConfig(BaseDataProcessorConfig):
    """Configuration for the PrecisionRecallFScoreSupport Processor."""

    beta: float = 1.0
    """The strength of recall versus precision in the F-score."""

    labels: list[int] | None = None
    """The set of labels to include and their order if average is None.
    Labels present in the data can be excluded, for example in multiclass classification
    to exclude a “negative class”. Labels not present in the data can be included and will
    be “assigned” 0 samples. For multilabel targets, labels are column indices.
    By default, all labels in y_true and y_pred are used in sorted order.
    """

    average: Literal["micro", "macro", "weighted"] | None = None
    """If None, the metrics for each class are returned. Otherwise, this determines
    the type of averaging performed on the data:

    `micro`:
    Calculate metrics globally by counting the total true positives, false negatives
    and false positives.

    `macro`:
    Calculate metrics for each label, and find their unweighted mean. This does not
    take label imbalance into account.

    `weighted`:
    Calculate metrics for each label, and find their average weighted by support
    (the number of true instances for each label). This alters `macro` to account for
    label imbalance; it can result in an F-score that is not between precision and recall.
    """

    warn_for: list | tuple | set = ("precision", "recall", "f-score")
    """For internal use.
    
    This determines which warnings will be made in the case that this function is
    being used to return only one of its metrics.
    """

    zero_division: Literal["warn"] | float = "warn"
    """Sets the value to return when there is a zero division:

    {“warn”, 0.0, 1.0, np.nan}, default=”warn”

    - recall: when there are no positive labels
    - precision: when there are no positive predictions
    - f-score: both

    Notes: - If set to “warn”, this acts like 0, but a warning is also raised.
    If set to np.nan, such values will be excluded from the average.
    """


def check_input_confusion_matrix(
    config: PrecisionRecallFScoreSupportConfig, ref: FeatureRef
) -> None:
    """Validates the input confusion matrix against the processor configuration.

    This function ensures that the input confusion matrix has the correct feature type
    (Array3D) and that the length of the labels specified in the configuration does not
    exceed the number of classes in the confusion matrix. It raises a RuntimeError if
    the labels are longer than the number of classes.

    Args:
        config (PrecisionRecallFScoreSupportConfig): Configuration parameters for the processor.
        ref (FeatureRef): Reference to the input feature, which should be a confusion matrix.

    Raises:
        RuntimeError: If the length of the labels in the configuration is greater than the
        number of classes in the confusion matrix.
    """
    shape = get_sequence_shape(ref.feature_)
    assert shape[1:] == (2, 2)

    # check labels argument
    if config.labels is not None and len(config.labels) > shape[0]:
        raise RuntimeError(
            "Labels in `PrecisionRecallFScoreSupportConfig` cannot longer than the number "
            "of classes in the confusion matrix. Confusion matrix has shape "
            f"{shape}, but labels have length {len(config.labels)}."
        )


class PrecisionRecallFScoreSupportInputRefs(InputRefs):
    """Input ref description for the PrecisionRecallFScoreSupport Processor."""

    confusion_matrix: Annotated[
        FeatureRef, FeatureValidator(check_input_confusion_matrix)
    ]
    """Confusion matrix to compute scores from.
    
    Must be a nested Sequence of shape (n_classes, 2, 2)
    """

    # TODO: sample_weight


def infer_prfs_output_feature(
    config: PrecisionRecallFScoreSupportConfig,
    inputs: PrecisionRecallFScoreSupportInputRefs,
    dtype: str,
) -> FeatureType:
    """Infers the feature type for precision, recall, F-score and support outputs.

    Args:
        config (PrecisionRecallFScoreSupportConfig): Configuration parameters for the processor.
        inputs (PrecisionRecallFScoreSupportInputRefs): Input references for the processor.
        dtype (str): The dtype of the output feature.

    Returns:
        FeatureType: The inferred feature type for precision, recall, and F-score outputs.
    """
    if config.average is not None:
        return Value(dtype)
    elif config.labels is not None:
        return Sequence(Value(dtype), length=len(config.labels))
    else:
        return Sequence(
            Value(dtype),
            length=get_sequence_length(inputs["confusion_matrix"].feature_),
        )


class PrecisionRecallFScoreSupportOutputRefs(OutputRefs):
    """Outputs of the PrecisionRecallFScoreSupport Processor."""

    precision: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: infer_prfs_output_feature(c, i, "float32")
        ),
    ]
    """Precision score.
    
    float (if average is not None) or sequence of float, shape = [n_unique_labels]
    """

    recall: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: infer_prfs_output_feature(c, i, "float32")
        ),
    ]
    """Recall score.

    float (if average is not None) or sequence of float, shape = [n_unique_labels]
    """

    f_score: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: infer_prfs_output_feature(c, i, "float32")
        ),
    ]
    """F-beta score.
    
    float (if average is not None) or sequence of float, shape = [n_unique_labels]
    """

    support: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, i: infer_prfs_output_feature(c, i, "int64")
        ),
    ]
    """The number of occurrences of each label in y_true. If average is not None, this
    will be computed as the sum of each label occurences in y_true.
    
    sequence of int, shape = [n_unique_labels]
    """


class PrecisionRecallFScoreSupport(
    BaseDataProcessor[
        PrecisionRecallFScoreSupportConfig,
        PrecisionRecallFScoreSupportInputRefs,
        PrecisionRecallFScoreSupportOutputRefs,
    ]
):
    """The :class`PrecisionRecallFScoreSupport` Processor.

    Implements an adaption of sklearn's :code:`precision_recall_fscore_support` function,
    but takes the confusion matrix as an input.
    """

    async def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        """Processes a single input sample and returns the corresponding output sample.

        Args:
            inputs (Sample): The input sample to be processed.
            index (int): The index associated with the input sample.
            rank (int): The rank of the processor in a distributed setting.
            io (IOContext): Context information for the data processors execution.

        Returns:
            Sample: The processed output sample.
        """
        # config parameters
        average = self.config.average
        beta = self.config.beta
        labels = self.config.labels
        warn_for = self.config.warn_for
        zero_division = self.config.zero_division

        # input confusion matrix
        MCM = np.array(inputs["confusion_matrix"])
        # reduce the confusion matrix to selected labels
        if labels is not None:
            MCM = MCM[labels]

        # *** The code below is mostly kläut by sklearn's `precision_recall_fscore_support` ***
        # The only difference is that we start with the confusion matrix
        tp_sum = MCM[:, 1, 1]
        pred_sum = tp_sum + MCM[:, 0, 1]
        true_sum = tp_sum + MCM[:, 1, 0]

        if average == "micro":
            tp_sum = np.array([tp_sum.sum()])
            pred_sum = np.array([pred_sum.sum()])
            true_sum = np.array([true_sum.sum()])

        # Finally, we have all our sufficient statistics. Divide! #
        beta2 = beta**2

        # Divide, and on zero-division, set scores and/or warn according to
        # zero_division:
        precision = _prf_divide(
            tp_sum,
            pred_sum,
            "precision",
            "predicted",
            average,
            warn_for,
            zero_division,
        )
        recall = _prf_divide(
            tp_sum,
            true_sum,
            "recall",
            "true",
            average,
            warn_for,
            zero_division,
        )

        if np.isposinf(beta):
            f_score = recall
        elif beta == 0:
            f_score = precision
        else:
            # The score is defined as:
            # score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            # Therefore, we can express the score in terms of confusion matrix entries as:
            # score = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)
            denom = beta2 * true_sum + pred_sum
            f_score = _prf_divide(
                (1 + beta2) * tp_sum,
                denom,
                "f-score",
                "true nor predicted",
                average,
                warn_for,
                zero_division,
            )

        # Average the results
        if average == "weighted":
            weights = true_sum
        else:
            weights = None

        if average is not None:
            assert average != "binary" or len(precision) == 1
            precision = _nanaverage(precision, weights=weights)
            recall = _nanaverage(recall, weights=weights)
            f_score = _nanaverage(f_score, weights=weights)
            # Different to sklearn, we return the support as the sum of
            # all target labels support values, if average is not None
            true_sum = true_sum.sum()

        return_sample = Sample(
            precision=precision.tolist(),
            recall=recall.tolist(),
            f_score=f_score.tolist(),
            support=true_sum.astype("int").tolist(),
        )

        return return_sample

    def call(
        self, **kwargs: Unpack[PrecisionRecallFScoreSupportInputRefs]
    ) -> PrecisionRecallFScoreSupportOutputRefs:
        """Adds the processor to the data flow.

        This method first prepares the inputs, then adds the processor to the data
        flow and returns a feature reference to the output features of the processor.

        Args:
            confusion_matrix (FeatureRef): The input confusion matrix from which
                to compute the scores.
            **kwargs (FeatureRef): Keyword arguments passed to call method.

        Returns:
            PrecisionRecallFScoreSupportOutputRefs: The output references produced by the processor.
        """
        return super().call(**kwargs)
