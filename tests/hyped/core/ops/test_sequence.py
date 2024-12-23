from hyped.core.ops.sequence import (
    SequenceGetItem,
    SequenceGetSlice,
    SequenceLength,
    SequenceMax,
    SequenceMin,
    SequenceSum,
)
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.typing import Bool, Int, Sequence


class TestStringAdd(BaseDataProcessorTest):
    processor = SequenceLength()
    input_features = {"x": Sequence[Bool]}
    input_data = [{"x": [False, False, False]}, {"x": [True, True]}]
    expected_output_feature = Int
    expected_output_data = [3, 2]


class TestStringMin(BaseDataProcessorTest):
    processor = SequenceMin()
    input_features = {"x": Sequence[Int]}
    input_data = [{"x": [2, 1, 0]}, {"x": [5, 3, 7]}]
    expected_output_feature = Int
    expected_output_data = [0, 3]


class TestSequenceMax(BaseDataProcessorTest):
    processor = SequenceMax()
    input_features = {"x": Sequence[Int]}
    input_data = [{"x": [2, 1, 0]}, {"x": [5, 3, 7]}]  # Maximum value is 2  # Maximum value is 7
    expected_output_feature = Int
    expected_output_data = [2, 7]


class TestSequenceSum(BaseDataProcessorTest):
    processor = SequenceSum()
    input_features = {"x": Sequence[Int]}
    input_data = [{"x": [2, 1, 0]}, {"x": [5, 3, 7]}]  # Sum is 3  # Sum is 15
    expected_output_feature = Int
    expected_output_data = [3, 15]


class TestSequenceGetItem(BaseDataProcessorTest):
    processor = SequenceGetItem(index=1)  # Retrieving the second item in the sequence
    input_features = {"sequence": Sequence[Int]}
    input_data = [
        {"sequence": [2, 1, 0]},  # Item at index 1 is 1
        {"sequence": [5, 3, 7]},  # Item at index 1 is 3
    ]
    expected_output_feature = Int
    expected_output_data = [1, 3]


class TestSequenceGetSlice(BaseDataProcessorTest):
    processor = SequenceGetSlice(
        start=1, stop=3, step=1
    )  # Slice from index 1 to 3 (exclusive) with step 1
    input_features = {"sequence": Sequence[Int]}
    input_data = [
        {"sequence": [2, 1, 0]},  # Slice [1, 0]
        {"sequence": [5, 3, 7, 9]},  # Slice [3, 7]
    ]
    expected_output_feature = Sequence[Int]
    expected_output_data = [[1, 0], [3, 7]]
