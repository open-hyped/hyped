from datasets import Features, Value

from hyped.data.flow.processors.ops.value import any_value
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestEquals(BaseDataProcessorTest):
    processor_type = any_value.Equals
    processor_config = any_value.EqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [True, False, False]}


class TestNotEquals(BaseDataProcessorTest):
    processor_type = any_value.NotEquals
    processor_config = any_value.NotEqualsConfig()

    input_features = Features({"a": Value("int32"), "b": Value("int32")})
    input_data = {"a": [0, 0, 1], "b": [0, 1, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}
