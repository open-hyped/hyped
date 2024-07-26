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
