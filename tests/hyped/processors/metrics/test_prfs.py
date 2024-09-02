from datasets import Features, Sequence, Value

from hyped.processors.metrics.prfs import (
    PrecisionRecallFScoreSupport,
    PrecisionRecallFScoreSupportConfig,
)
from tests.hyped.processors.base import BaseDataProcessorTest


class TestPRFS(BaseDataProcessorTest):
    processor_type = PrecisionRecallFScoreSupport
    processor_config = PrecisionRecallFScoreSupportConfig()

    input_features = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )
    input_data = {"confusion_matrix": [[[[3, 0], [0, 1]], [[3, 0], [0, 1]], [[2, 0], [0, 2]]]]}
    input_index = [0]

    expected_output_features = Features(
        {
            "precision": Sequence(Value("float32"), length=3),
            "recall": Sequence(Value("float32"), length=3),
            "f_score": Sequence(Value("float32"), length=3),
            "support": Sequence(Value("int64"), length=3),
        }
    )
    expected_output_data = {
        "precision": [[1.0, 1.0, 1.0]],
        "recall": [[1.0, 1.0, 1.0]],
        "f_score": [[1.0, 1.0, 1.0]],
        "support": [[1, 1, 2]],
    }


class TestPRFSMicro(BaseDataProcessorTest):
    processor_type = PrecisionRecallFScoreSupport
    processor_config = PrecisionRecallFScoreSupportConfig(average="micro")

    input_features = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )
    input_data = {"confusion_matrix": [[[[3, 0], [0, 1]], [[3, 0], [0, 1]], [[2, 0], [0, 2]]]]}
    input_index = [0]

    expected_output_features = Features(
        {
            "precision": Value("float32"),
            "recall": Value("float32"),
            "f_score": Value("float32"),
            "support": Value("int64"),
        }
    )
    expected_output_data = {
        "precision": [1.0],
        "recall": [1.0],
        "f_score": [1.0],
        "support": [4],
    }


class TestPRFSMacro(BaseDataProcessorTest):
    processor_type = PrecisionRecallFScoreSupport
    processor_config = PrecisionRecallFScoreSupportConfig(average="macro")

    input_features = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )
    input_data = {"confusion_matrix": [[[[3, 0], [0, 1]], [[3, 0], [0, 1]], [[2, 0], [0, 2]]]]}
    input_index = [0]

    expected_output_features = Features(
        {
            "precision": Value("float32"),
            "recall": Value("float32"),
            "f_score": Value("float32"),
            "support": Value("int64"),
        }
    )
    expected_output_data = {
        "precision": [1.0],
        "recall": [1.0],
        "f_score": [1.0],
        "support": [4],
    }


class TestPRFSWeighted(BaseDataProcessorTest):
    processor_type = PrecisionRecallFScoreSupport
    processor_config = PrecisionRecallFScoreSupportConfig(average="weighted")

    input_features = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )
    input_data = {"confusion_matrix": [[[[3, 0], [0, 1]], [[3, 0], [0, 1]], [[2, 0], [0, 2]]]]}
    input_index = [0]

    expected_output_features = Features(
        {
            "precision": Value("float32"),
            "recall": Value("float32"),
            "f_score": Value("float32"),
            "support": Value("int64"),
        }
    )
    expected_output_data = {
        "precision": [1.0],
        "recall": [1.0],
        "f_score": [1.0],
        "support": [4],
    }


class TestPRFSBetaZero(BaseDataProcessorTest):
    processor_type = PrecisionRecallFScoreSupport
    processor_config = PrecisionRecallFScoreSupportConfig(beta=0)

    input_features = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )
    input_data = {"confusion_matrix": [[[[3, 0], [0, 1]], [[3, 0], [0, 1]], [[2, 0], [0, 2]]]]}
    input_index = [0]

    expected_output_features = Features(
        {
            "precision": Sequence(Value("float32"), length=3),
            "recall": Sequence(Value("float32"), length=3),
            "f_score": Sequence(Value("float32"), length=3),
            "support": Sequence(Value("int64"), length=3),
        }
    )
    expected_output_data = {
        "precision": [[1.0, 1.0, 1.0]],
        "recall": [[1.0, 1.0, 1.0]],
        "f_score": [[1.0, 1.0, 1.0]],
        "support": [[1, 1, 2]],
    }


class TestPRFSWithLabels(BaseDataProcessorTest):
    processor_type = PrecisionRecallFScoreSupport
    processor_config = PrecisionRecallFScoreSupportConfig(labels=(0, 2))

    input_features = Features(
        {
            "confusion_matrix": Sequence(
                Sequence(Sequence(Value("int64"), length=2), length=2),
                length=3,
            )
        }
    )
    input_data = {"confusion_matrix": [[[[3, 0], [0, 1]], [[3, 0], [0, 1]], [[2, 0], [0, 2]]]]}
    input_index = [0]

    expected_output_features = Features(
        {
            "precision": Sequence(Value("float32"), length=2),
            "recall": Sequence(Value("float32"), length=2),
            "f_score": Sequence(Value("float32"), length=2),
            "support": Sequence(Value("int64"), length=2),
        }
    )
    expected_output_data = {
        "precision": [[1.0, 1.0]],
        "recall": [[1.0, 1.0]],
        "f_score": [[1.0, 1.0]],
        "support": [[1, 2]],
    }
