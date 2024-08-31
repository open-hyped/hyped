from datasets import ClassLabel, Features, Sequence, Value

from hyped.data.flow.processors.utils.to_class_label import ToClassLabel
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestToClassLabel(BaseDataProcessorTest):
    processor_type = ToClassLabel
    processor_config = ToClassLabel.Config(class_labels=["A", "B"])

    input_features = Features({"label": Value("string")})
    input_data = {"label": ["A", "A", "B", "B", "A"]}

    expected_output_features = Features(
        {"class_label": ClassLabel(names=["A", "B"])}
    )
    expected_output_data = {"class_label": [0, 0, 1, 1, 0]}


class TestToClassLabelWithSequences(BaseDataProcessorTest):
    processor_type = ToClassLabel
    processor_config = ToClassLabel.Config(class_labels=["A", "B"])

    input_features = Features({"label": Sequence(Value("string"))})
    input_data = {"label": [["A", "B"], ["B", "A", "A"], ["A"], []]}

    expected_output_features = Features(
        {"class_label": Sequence(ClassLabel(names=["A", "B"]))}
    )
    expected_output_data = {"class_label": [[0, 1], [1, 0, 0], [0], []]}


class TestToClassLabelWithFixedLengthSequences(BaseDataProcessorTest):
    processor_type = ToClassLabel
    processor_config = ToClassLabel.Config(class_labels=["A", "B"])

    input_features = Features({"label": Sequence(Value("string"), length=3)})
    input_data = {
        "label": [
            ["A", "B", "A"],
            ["B", "A", "B"],
            ["A", "A", "A"],
            ["B", "B", "B"],
        ]
    }

    expected_output_features = Features(
        {"class_label": Sequence(ClassLabel(names=["A", "B"]), length=3)}
    )
    expected_output_data = {
        "class_label": [[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]]
    }
