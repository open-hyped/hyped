"""Provides metric operations on feature references.

This module provides functions for calculating performance metrics for
classification tasks, specifically focusing on precision, recall, F-score, 
and support. These metrics are essential in evaluating the effectiveness of 
a classification model, particularly in scenarios involving multi-class or 
multi-label classification.

Key functionalities include:

- **Precision, Recall, F-Score, and Support Calculation** 
  (`precision_recall_fscore_support`): Compute precision, recall, F-score, 
  and support for given ground truths and predictions across different 
  classification tasks. The function handles both multi-class and multi-label 
  classification scenarios.

- **Multi-Class Classification**: When provided with class labels as 
  input features, the function interprets the task as a multi-class 
  classification problem, computing metrics based on the assumption 
  that each instance belongs to exactly one class.

- **Multi-Label Classification**: When provided with sequences of boolean 
  values representing class presence or absence, the function interprets the 
  task as a multi-label classification problem, where each instance may belong 
  to multiple classes.
"""
from typing import Literal

import hyped.aggregators as aggregators
from hyped.core.refs.ref import FeatureRef
from hyped.processors.metrics.prfs import (
    PrecisionRecallFScoreSupport,
    PrecisionRecallFScoreSupportOutputRefs,
)


def precision_recall_fscore_support(
    y_true: FeatureRef,
    y_pred: FeatureRef,
    labels: list[int] | None = None,
    beta: float = 1.0,
    average: Literal["micro", "macro", "weighted"] | None = None,
    warn_for: list | tuple | set = ("precision", "recall", "f-score"),
    zero_division: Literal["warn"] | float = "warn",
) -> PrecisionRecallFScoreSupportOutputRefs:
    """Compute precision, recall, f-score and support for given predictions and ground truths.

    This function takes two inputs, namely :class:`y_true`and :class:`y_pred`, which can
    take on different forms indicating the assumed classification task:

    1. **Multi-class Classification:**
       - If :class:`y_true`and :class:`y_pred` is a single class label, the task is interpreted as
       a standard multi-class classification problem. In this setting, each instance is assigned
       exactly one class label out of the possible :code:`n_classes` labels.

    2. **Multi-label Classification:**
       - If :class:`y_true`and :class:`y_pred` are a sequences of boolean values with shape
       :code:`(n_classes,)`,  then the task is interpreted as a multi-label classification problem.
       Here, each class label is represented as a boolean value, where :code:`True` indicates the
       presence of the class,  and :code:`False` indicates its absence. Multiple classes can be
       associated with a single instance.

    Args:
        y_true (FeatureRef): The ground truths. Either a class label feature in case of
            standard multi-class classification, or a sequence of boolean values in case
            of multi-label classification.
        y_pred (FeatureRef): The predictions. Either a class label feature in case of
            standard multi-class classification, or a sequence of boolean values in case
            of multi-label classification.
        labels (list): The set of labels to include and their order if average is None.
            Labels present in the data can be excluded, for example in multiclass classification
            to exclude a “negative class”. Labels not present in the data can be included and
            will be “assigned” 0 samples. For multilabel targets, labels are column indices.
            By default, all labels in y_true and y_pred are used in sorted order.
        beta (float): The strength of recall versus precision in the F-score.
        average (str): If None, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            `micro`:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.

            `macro`:
            Calculate metrics for each label, and find their unweighted mean. This does
            not take label imbalance into account.

            `weighted`:
            Calculate metrics for each label, and find their average weighted by support
            (the number of true instances for each label). This alters `macro` to account for
            label imbalance; it can result in an F-score that is not between precision and recall.
        warn_for (list | tuple | set): For internal use. This determines which warnings will be
            made in the case that this function is being used to return only one of its metrics.
        zero_division: (str | float): Sets the value to return when there is a zero division:

            {“warn”, 0.0, 1.0, np.nan}, default=”warn”

            - recall: when there are no positive labels
            - precision: when there are no positive predictions
            - f-score: both

            Notes: - If set to “warn”, this acts like 0, but a warning is also raised.
            If set to np.nan, such values will be excluded from the average.

    Returns:
        FeatureRef: A :class:`FeatureRef` instance representing
        the aggregated scores.

    Raises:
        TypeError: If the features are of unexpected types.
    """
    multilabel_confusion_matrix = (
        aggregators.MultiLabelConfusionMatrix()
        .call(
            y_true=y_true,
            y_pred=y_pred,
        )
        .confusion_matrix
    )

    return PrecisionRecallFScoreSupport(
        labels=labels,
        beta=beta,
        average=average,
        warn_for=warn_for,
        zero_division=zero_division,
    ).call(confusion_matrix=multilabel_confusion_matrix)
