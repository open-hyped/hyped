from datasets import Features, Sequence, Value

from hyped.data.flow.processors.utils.bool_index import BooleanIndexing
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestBooleanIndexing(BaseDataProcessorTest):
    processor_type = BooleanIndexing
    processor_config = BooleanIndexing.Config()

    input_features = Features(
        {"values": Sequence(Value("int32")), "mask": Sequence(Value("bool"))}
    )
    input_data = {
        "values": [[1, 2, 3, 4], [10, 20, 30]],
        "mask": [[True, False, True, False], [False, True, True]],
    }

    expected_output_features = Features(
        {"indexed_values": Sequence(Value("int32"), length=-1)}
    )
    expected_output_data = {"indexed_values": [[1, 3], [20, 30]]}


class TestBooleanIndexingWithEmptySequences(BaseDataProcessorTest):
    processor_type = BooleanIndexing
    processor_config = BooleanIndexing.Config()

    input_features = Features(
        {"values": Sequence(Value("int32")), "mask": Sequence(Value("bool"))}
    )
    input_data = {
        "values": [[], [1, 2, 3]],
        "mask": [[], [False, False, False]],
    }

    expected_output_features = Features(
        {"indexed_values": Sequence(Value("int32"), length=-1)}
    )
    expected_output_data = {"indexed_values": [[], []]}


class TestBooleanIndexingWithStrings(BaseDataProcessorTest):
    processor_type = BooleanIndexing
    processor_config = BooleanIndexing.Config()

    input_features = Features(
        {"values": Sequence(Value("string")), "mask": Sequence(Value("bool"))}
    )
    input_data = {
        "values": [["apple", "banana", "cherry"], ["dog", "cat", "mouse"]],
        "mask": [[True, False, True], [True, True, False]],
    }

    expected_output_features = Features(
        {"indexed_values": Sequence(Value("string"), length=-1)}
    )
    expected_output_data = {
        "indexed_values": [["apple", "cherry"], ["dog", "cat"]]
    }


class TestBooleanIndexingWithFixedLengthSequences(BaseDataProcessorTest):
    processor_type = BooleanIndexing
    processor_config = BooleanIndexing.Config()

    input_features = Features(
        {
            "values": Sequence(Value("float32"), length=3),
            "mask": Sequence(Value("bool"), length=3),
        }
    )
    input_data = {
        "values": [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]],
        "mask": [
            [True, False, True],
            [False, True, False],
            [True, True, True],
        ],
    }

    expected_output_features = Features(
        {"indexed_values": Sequence(Value("float32"), length=-1)}
    )
    expected_output_data = {
        "indexed_values": [[1.1, 3.3], [5.5], [7.7, 8.8, 9.9]]
    }
