from hyped.core.ops.boolean import And, Invert, Or, Xor
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Bool


class TestInvert(BaseDataProcessorTest):
    processor = Invert()
    input_features = {"x": Bool}
    input_data = [{"x": True}, {"x": False}]
    expected_output_feature = Bool
    expected_output_data = [False, True]


class TestAnd(BaseDataProcessorTest):
    processor = And()
    input_features = {"a": Bool, "b": Bool}
    input_data = [
        {"a": True, "b": True},
        {"a": True, "b": False},
        {"a": False, "b": True},
        {"a": False, "b": False},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, False, False, False]


class TestOr(BaseDataProcessorTest):
    processor = Or()
    input_features = {"a": Bool, "b": Bool}
    input_data = [
        {"a": True, "b": True},
        {"a": True, "b": False},
        {"a": False, "b": True},
        {"a": False, "b": False},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, True, True, False]


class TestXOr(BaseDataProcessorTest):
    processor = Xor()
    input_features = {"a": Bool, "b": Bool}
    input_data = [
        {"a": True, "b": True},
        {"a": True, "b": False},
        {"a": False, "b": True},
        {"a": False, "b": False},
    ]
    expected_output_feature = Bool
    expected_output_data = [False, True, True, False]
