from hyped.core.ops.comparison import Equals
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.core.typing import Bool, Float, Int, Mapping, Sequence, String


class TestEqualsBoolBool(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Bool, "obj2": Bool}
    input_data = [
        # True
        {"obj1": True, "obj2": True},
        {"obj1": False, "obj2": False},
        # False
        {"obj1": False, "obj2": True},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, True, False]


class TestEqualsIntInt(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Int, "obj2": Int}
    input_data = [
        # True
        {"obj1": 1, "obj2": 1},
        {"obj1": -1, "obj2": -1},
        {"obj1": 0, "obj2": 0},
        # False
        {"obj1": 0, "obj2": 1},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, True, True, False]


class TestEqualsFloatFloat(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Float, "obj2": Float}
    input_data = [
        # True
        {"obj1": 5e-10, "obj2": 5e-10},
        {"obj1": 1 / 3, "obj2": 3 / 9},
        # False
        {"obj1": 0.0, "obj2": 1.0},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, True, False]


class TestEqualsStringString(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": String, "obj2": String}
    input_data = [
        # True
        {"obj1": "Hello", "obj2": "Hello"},
        # False
        {"obj1": "Hello", "obj2": "olleH"},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, False]


class TestEqualsSequenceSequence(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Sequence[Int], "obj2": Sequence[Int]}
    input_data = [
        # True
        {"obj1": [0, 1, 2], "obj2": [0, 1, 2]},
        # False
        {"obj1": [1, 2, 3], "obj2": [0, 1, 2]},
        {"obj1": [0, 1, 2, 3], "obj2": [0, 1, 2]},
    ]
    expected_output_feature = Bool
    expected_output_data = [True, False, False]


class NestedMap(Mapping):
    x: String
    y: Sequence[Int]


class NestedMapMatch(Mapping):
    x: String
    y: Sequence[Int]


class NestedMapMismatch(Mapping):
    x: String
    y: Sequence[Float]


class Map(Mapping):
    a: Int
    b: NestedMap


class MapMatch(Mapping):
    a: Int
    b: NestedMap


class MapMatchNested(Mapping):
    a: Int
    b: NestedMapMatch


class MapMismatchNested(Mapping):
    a: Int
    b: NestedMapMismatch


class MapMismatch(Mapping):
    a: String
    b: Float


class TestEqualsMapMapMatch(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Map, "obj2": MapMatch}
    input_data = [
        # True
        {
            "obj1": {"a": 0, "b": {"x": "A", "y": [0, 1]}},
            "obj2": {"a": 0, "b": {"x": "A", "y": [0, 1]}},
        },
        # False
        {
            "obj1": {"a": 0, "b": {"x": "A", "y": [0, 1]}},
            "obj2": {"a": 0, "b": {"x": "A", "y": [1, 0]}},
        },
    ]
    expected_output_feature = Bool
    expected_output_data = [True, False]


class TestEqualsMapMapMatchNested(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Map, "obj2": MapMatchNested}
    input_data = [
        # True
        {
            "obj1": {"a": 0, "b": {"x": "A", "y": [0, 1]}},
            "obj2": {"a": 0, "b": {"x": "A", "y": [0, 1]}},
        },
        # False
        {
            "obj1": {"a": 0, "b": {"x": "A", "y": [0, 1]}},
            "obj2": {"a": 0, "b": {"x": "A", "y": [1, 0]}},
        },
    ]
    expected_output_feature = Bool
    expected_output_data = [True, False]


class TestEqualsMapMapMismatch(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Map, "obj2": MapMismatch}
    expected_verification_error = TypeError


class TestEqualsMapMapMismatchNested(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Map, "obj2": MapMismatchNested}
    expected_verification_error = TypeError


class TestEqualsTypeMismatch(BaseDataProcessorTest):
    processor = Equals()
    input_features = {"obj1": Bool, "obj2": Int}
    expected_verification_error = TypeError
