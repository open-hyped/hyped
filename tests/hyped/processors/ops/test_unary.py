from datasets import Features, Sequence, Value

from hyped.processors._ops import unary
from tests.hyped.processors.base import BaseDataProcessorTest


class TestNeg(BaseDataProcessorTest):
    processor_type = unary.Neg
    processor_config = unary.NegConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [-1, 2, 0]}


class TestInvert(BaseDataProcessorTest):
    processor_type = unary.Invert
    processor_config = unary.InvertConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [-2, 1, -1]}


class TestAbs(BaseDataProcessorTest):
    processor_type = unary.Abs
    processor_config = unary.AbsConfig()

    input_features = Features({"a": Value("int32")})
    input_data = {"a": [1, -2, 0]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("int32")})
    expected_output_data = {"result": [1, 2, 0]}


class TestBooleanInvert(BaseDataProcessorTest):
    processor_type = unary.BooleanInvert
    processor_config = unary.BooleanInvertConfig()

    input_features = Features({"a": Value("bool")})
    input_data = {"a": [True, False, True]}
    input_index = [0, 1, 2]

    expected_output_features = Features({"result": Value("bool")})
    expected_output_data = {"result": [False, True, False]}


class TestElementWiseNeg(BaseDataProcessorTest):
    processor_type = unary.Neg
    processor_config = unary.NegConfig()

    input_features = Features({"a": Sequence(Value("int32"))})
    input_data = {"a": [[1, 2, 3], [1, 2]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int32"))})
    expected_output_data = {"result": [[-1, -2, -3], [-1, -2]]}


class TestElementWiseAbs(BaseDataProcessorTest):
    processor_type = unary.Abs
    processor_config = unary.AbsConfig()

    input_features = Features({"a": Sequence(Value("float32"))})
    input_data = {"a": [[1, -2, 3], [-1, -2]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("float32"))})
    expected_output_data = {"result": [[1, 2, 3], [1, 2]]}


class TestElementWiseInvert(BaseDataProcessorTest):
    processor_type = unary.Invert
    processor_config = unary.InvertConfig()

    input_features = Features({"a": Sequence(Value("int64"))})
    input_data = {"a": [[1, 2, 3], [-1, -2, -3]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("int64"))})
    expected_output_data = {"result": [[-2, -3, -4], [0, 1, 2]]}


class TestElementWiseBooleanInvert(BaseDataProcessorTest):
    processor_type = unary.BooleanInvert
    processor_config = unary.BooleanInvertConfig()

    input_features = Features({"a": Sequence(Value("bool"))})
    input_data = {"a": [[True, False, False], [False, True, False]]}
    input_index = [0, 1]

    expected_output_features = Features({"result": Sequence(Value("bool"))})
    expected_output_data = {
        "result": [[False, True, True], [True, False, True]]
    }
