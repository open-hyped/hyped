from datasets import Features, Value

from hyped.data.flow.processors.ops.value import boolean
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestInvert(BaseDataProcessorTest):
    processor_type = boolean.BooleanInvert
    processor_config = boolean.BooleanInvertConfig()

    input_features = Features({"a": Value("bool")})
    input_data = {"a": [True, False, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestLogicalAnd(BaseDataProcessorTest):
    processor_type = boolean.LogicalAnd
    processor_config = boolean.LogicalAndConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, False, True]}


class TestLogicalOr(BaseDataProcessorTest):
    processor_type = boolean.LogicalOr
    processor_config = boolean.LogicalOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, True]}


class TestLogicalXOr(BaseDataProcessorTest):
    processor_type = boolean.LogicalXOr
    processor_config = boolean.LogicalXOrConfig()

    input_features = Features({"a": Value("bool"), "b": Value("bool")})
    input_data = {"a": [False, False, True], "b": [False, True, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}
