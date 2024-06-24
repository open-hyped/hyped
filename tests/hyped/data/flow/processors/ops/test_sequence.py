from datasets import Features, Sequence, Value

from hyped.data.flow.processors.ops import sequence
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestComparator(BaseDataProcessorTest):
    processor_type = sequence.Concat
    processor_config = sequence.ConcatConfig()

    input_features = Features(
        {"a": Sequence(Value("int32")), "b": Sequence(Value("int32"))}
    )
    input_data = {"a": [[1, 2, 3]], "b": [[4, 5, 6]]}
    input_index = [0]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[1, 2, 3, 4, 5, 6]]}


class TestComparator_FixedLength(BaseDataProcessorTest):
    processor_type = sequence.Concat
    processor_config = sequence.ConcatConfig()

    input_features = Features(
        {
            "a": Sequence(Value("int32"), length=3),
            "b": Sequence(Value("int32"), length=3),
        }
    )
    input_data = {"a": [[1, 2, 3]], "b": [[4, 5, 6]]}
    input_index = [0]

    expected_output_features = Features(
        {"result": Sequence(Value("int32"), length=6)}
    )
    expected_output_data = {"result": [[1, 2, 3, 4, 5, 6]]}
