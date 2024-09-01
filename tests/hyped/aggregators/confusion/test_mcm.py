import numpy as np
from datasets import ClassLabel, Features, Sequence, Value

from hyped.aggregators.confusion.mcm import (
    MultiLabelConfusionMatrix,
    MultiLabelConfusionMatrixConfig,
)
from tests.hyped.aggregators.base import BaseDataAggregatorTest


class TestMultiLabelConfusionMatrix(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MultiLabelConfusionMatrix
    aggregator_config = MultiLabelConfusionMatrixConfig()

    # input
    input_features = Features(
        {
            "y_true": Sequence(Value("bool"), length=3),
            "y_pred": Sequence(Value("bool"), length=3),
        }
    )

    input_data = {
        "y_true": [
            [True, False, True],
            [False, True, False],
        ],
        "y_pred": [
            [True, False, False],
            [False, True, True],
        ],
    }
    input_index = list(range(2))

    expected_value_feature = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )

    # expected initial value
    expected_initial_value = {
        "confusion_matrix": np.zeros((3, 2, 2), dtype=np.int64).tolist()
    }

    # expected output
    expected_output_value = {
        "confusion_matrix": [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
    }
    expected_output_state = None


class TestMultiLabelConfusionMatrixWithLabel(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MultiLabelConfusionMatrix
    aggregator_config = MultiLabelConfusionMatrixConfig(labels=(0, 2))

    # input
    input_features = Features(
        {
            "y_true": Sequence(Value("bool"), length=3),
            "y_pred": Sequence(Value("bool"), length=3),
        }
    )

    input_data = {
        "y_true": [
            [True, False, True],
            [False, True, False],
        ],
        "y_pred": [
            [True, False, False],
            [False, True, True],
        ],
    }
    input_index = list(range(2))

    expected_value_feature = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=2,
            )
        }
    )

    # expected initial value
    expected_initial_value = {
        "confusion_matrix": np.zeros((2, 2, 2), dtype=np.int64).tolist()
    }

    # expected output
    expected_output_value = {
        "confusion_matrix": [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
    }
    expected_output_state = None


class TestMultiLabelConfusionMatrixWithClassLabel(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MultiLabelConfusionMatrix
    aggregator_config = MultiLabelConfusionMatrixConfig()

    # input
    input_features = Features(
        {
            "y_true": ClassLabel(names=["A", "B", "C"]),
            "y_pred": ClassLabel(names=["A", "B", "C"]),
        }
    )

    input_data = {
        "y_true": [0, 1, 1, 2, 1],
        "y_pred": [0, 0, 0, 2, 1],
    }
    input_index = list(range(5))

    expected_value_feature = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )

    # expected initial value
    expected_initial_value = {
        "confusion_matrix": np.zeros((3, 2, 2), dtype=np.int64).tolist()
    }

    # expected output
    expected_output_value = {
        "confusion_matrix": [
            [[2, 2], [0, 1]],
            [[2, 0], [2, 1]],
            [[4, 0], [0, 1]],
        ]
    }
    expected_output_state = None


class TestMultiLabelConfusionMatrixInvalidClassLabel(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MultiLabelConfusionMatrix
    aggregator_config = MultiLabelConfusionMatrixConfig()

    # input
    input_features = Features(
        {
            "y_true": ClassLabel(names=["A", "B", "C"]),
            "y_pred": ClassLabel(names=["1", "2", "3"]),
        }
    )

    expected_input_verification_error = RuntimeError


class TestMultiLabelConfusionMatrixInvalidSequences(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MultiLabelConfusionMatrix
    aggregator_config = MultiLabelConfusionMatrixConfig()

    # input
    input_features = Features(
        {
            "y_true": Sequence(Value("bool"), length=3),
            "y_pred": Sequence(Value("bool"), length=4),
        }
    )

    expected_input_verification_error = RuntimeError


class TestMultiLabelConfusionMatrixInvalidCombination(BaseDataAggregatorTest):
    # aggregator
    aggregator_type = MultiLabelConfusionMatrix
    aggregator_config = MultiLabelConfusionMatrixConfig()

    # input
    input_features = Features(
        {
            "y_true": ClassLabel(names=["A", "B", "C"]),
            "y_pred": Sequence(Value("bool"), length=3),
        }
    )

    expected_input_verification_error = RuntimeError
