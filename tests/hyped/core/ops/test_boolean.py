from hyped.core.ops.boolean import And, Invert, Or, Where, Xor
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Bool, Int


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


class TestWhere(BaseDataProcessorTest):
    processor = Where()
    input_features = {"cond": Bool, "a": Int, "b": Int}
    input_data = [
        {"cond": True, "a": 0, "b": 1},
        {"cond": False, "a": 0, "b": 1},
        {"cond": False, "a": 0, "b": 1},
        {"cond": True, "a": 0, "b": 1},
        {"cond": True, "a": 0, "b": 1},
        {"cond": False, "a": 0, "b": 1},
        {"cond": True, "a": 0, "b": 1},
    ]
    expected_output_feature = Int
    expected_output_data = [0, 1, 1, 0, 0, 1, 0]
