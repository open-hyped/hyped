from datasets import Features, Sequence, Value

from hyped.processors._ops.sequence import reduce
from tests.hyped.processors.base import BaseDataProcessorTest


class TestSequenceLength(BaseDataProcessorTest):
    processor_type = reduce.SequenceLength
    processor_config = reduce.SequenceLengthConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2], [1]]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int64")})
    expected_output_data = {"result": [3, 2, 1]}


class TestSequenceSum(BaseDataProcessorTest):
    processor_type = reduce.SequenceSum
    processor_config = reduce.SequenceSumConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2], [1]]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int64")})
    expected_output_data = {"result": [6, 3, 1]}


class TestSequenceMean(BaseDataProcessorTest):
    processor_type = reduce.SequenceMean
    processor_config = reduce.SequenceMeanConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2], [1]]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("float32")})
    expected_output_data = {"result": [2.0, 1.5, 1.0]}
