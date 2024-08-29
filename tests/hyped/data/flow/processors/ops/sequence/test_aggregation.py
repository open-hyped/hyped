from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops.sequence import aggregation
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestSequenceLength(BaseDataProcessorTest):
    processor_type = aggregation.SequenceLength
    processor_config = aggregation.SequenceLengthConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2], [1]]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int64")})
    expected_output_data = {"result": [3, 2, 1]}


class TestSequenceSum(BaseDataProcessorTest):
    processor_type = aggregation.SequenceSum
    processor_config = aggregation.SequenceSumConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2], [1]]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int64")})
    expected_output_data = {"result": [6, 3, 1]}


class TestSequenceMean(BaseDataProcessorTest):
    processor_type = aggregation.SequenceMean
    processor_config = aggregation.SequenceMeanConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2], [1]]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [2.0, 1.5, 1.0]}
