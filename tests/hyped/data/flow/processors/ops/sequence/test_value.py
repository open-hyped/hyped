from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops.sequence import value
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestSequenceContains(BaseDataProcessorTest):
    processor_type = value.SequenceContains
    processor_config = value.SequenceContainsConfig()

    input_features = Features({"sequence": Sequence(Value("int32")), "value": Value("int32")})
    input_data = {
        "sequence": [[0, 1, 2], [0, 0, 1], [0, 0, 0]],
        "value": [0, 1, 2],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"contains": Value("bool")})
    expected_output_data = {"contains": [True, True, False]}


class TestSequenceCountOf(BaseDataProcessorTest):
    processor_type = value.SequenceCountOf
    processor_config = value.SequenceCountOfConfig()

    input_features = Features({"sequence": Sequence(Value("int32")), "value": Value("int32")})
    input_data = {
        "sequence": [[0, 1, 2], [0, 0, 1], [0, 0, 0]],
        "value": [0, 0, 0],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"count": Value("int64")})
    expected_output_data = {"count": [1, 2, 3]}


class TestSequenceIndexOf(BaseDataProcessorTest):
    processor_type = value.SequenceIndexOf
    processor_config = value.SequenceIndexOfConfig()

    input_features = Features({"sequence": Sequence(Value("int32")), "value": Value("int32")})
    input_data = {
        "sequence": [[0, 1, 2], [0, 0, 1], [1, 0, 0]],
        "value": [0, 0, 0],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"index": Value("int64")})
    expected_output_data = {"index": [0, 0, 1]}
