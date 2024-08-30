from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops.sequence import query
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestSequenceContains(BaseDataProcessorTest):
    processor_type = query.SequenceContains
    processor_config = query.SequenceContainsConfig()

    input_features = Features(
        {"sequence": Sequence(Value("int32")), "value": Value("int32")}
    )
    input_data = {
        "sequence": [[0, 1, 2], [0, 0, 1], [0, 0, 0]],
        "value": [0, 1, 2],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, True, False]}


class TestSequenceCountOf(BaseDataProcessorTest):
    processor_type = query.SequenceCountOf
    processor_config = query.SequenceCountOfConfig()

    input_features = Features(
        {"sequence": Sequence(Value("int32")), "value": Value("int32")}
    )
    input_data = {
        "sequence": [[0, 1, 2], [0, 0, 1], [0, 0, 0]],
        "value": [0, 0, 0],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int64")})
    expected_output_data = {"result": [1, 2, 3]}


class TestSequenceIndexOf(BaseDataProcessorTest):
    processor_type = query.SequenceIndexOf
    processor_config = query.SequenceIndexOfConfig()

    input_features = Features(
        {"sequence": Sequence(Value("int32")), "value": Value("int32")}
    )
    input_data = {
        "sequence": [[0, 1, 2], [0, 0, 1], [1, 0, 0]],
        "value": [0, 0, 0],
    }
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int64")})
    expected_output_data = {"result": [0, 0, 1]}
