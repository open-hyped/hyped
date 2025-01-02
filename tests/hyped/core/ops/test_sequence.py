from hyped.core.ops.sequence import (
    SequenceGetItem,
    SequenceGetSlice,
    SequenceLength,
    SequenceMax,
    SequenceMin,
    SequencePack,
    SequenceSum,
    SequenceUnpack,
    SequenceUnpackWithIndex,
    SequenceValueWithIndex,
)
from hyped.core.testing.augmenter import BaseDataAugmenterTest
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.typing import Annotated, Bool, Int, Int32, Len, Sequence


class TestStringAdd(BaseDataProcessorTest):
    processor = SequenceLength()
    input_features = {"seq": Sequence[Bool]}
    input_data = [{"seq": [False, False, False]}, {"seq": [True, True]}]
    expected_output_feature = Int
    expected_output_data = [3, 2]


class TestStringMin(BaseDataProcessorTest):
    processor = SequenceMin()
    input_features = {"seq": Sequence[Int]}
    input_data = [{"seq": [2, 1, 0]}, {"seq": [5, 3, 7]}]
    expected_output_feature = Int
    expected_output_data = [0, 3]


class TestSequenceMax(BaseDataProcessorTest):
    processor = SequenceMax()
    input_features = {"seq": Sequence[Int]}
    input_data = [
        {"seq": [2, 1, 0]},  # Maximum value is 2
        {"seq": [5, 3, 7]},  # Maximum value is 7
    ]
    expected_output_feature = Int
    expected_output_data = [2, 7]


class TestSequenceSum(BaseDataProcessorTest):
    processor = SequenceSum()
    input_features = {"seq": Sequence[Int]}
    input_data = [{"seq": [2, 1, 0]}, {"seq": [5, 3, 7]}]  # Sum is 3  # Sum is 15
    expected_output_feature = Int
    expected_output_data = [3, 15]


class TestSequenceGetItem(BaseDataProcessorTest):
    processor = SequenceGetItem(index=1)  # Retrieving the second item in the sequence
    input_features = {"seq": Sequence[Int]}
    input_data = [
        {"seq": [2, 1, 0]},  # Item at index 1 is 1
        {"seq": [5, 3, 7]},  # Item at index 1 is 3
    ]
    expected_output_feature = Int
    expected_output_data = [1, 3]


class TestSequenceGetSlice(BaseDataProcessorTest):
    processor = SequenceGetSlice(
        start=1, stop=3, step=1
    )  # Slice from index 1 to 3 (exclusive) with step 1
    input_features = {"seq": Sequence[Int]}
    input_data = [
        {"seq": [2, 1, 0]},  # Slice [1, 0]
        {"seq": [5, 3, 7, 9]},  # Slice [3, 7]
    ]
    expected_output_feature = Sequence[Int]
    expected_output_data = [[1, 0], [3, 7]]


class TestSequenceUnpack(BaseDataAugmenterTest):
    augmenter = SequenceUnpack()
    input_features = {"seq": Sequence[Int]}
    input_data = [
        {"seq": [1, 2, 3]},
        {"seq": [4, 5, 6]},
    ]
    expected_output_feature = Int
    expected_output_data = [1, 2, 3, 4, 5, 6]


class TestSequenceUnpackWithIndex(BaseDataAugmenterTest):
    augmenter = SequenceUnpackWithIndex()
    input_features = {"seq": Sequence[Int]}
    input_data = [
        {"seq": [1, 2, 3]},
        {"seq": [4, 5, 6]},
    ]
    expected_output_feature = SequenceValueWithIndex[Int]
    expected_output_data = [
        {"value": 1, "index": 0},
        {"value": 2, "index": 0},
        {"value": 3, "index": 0},
        {"value": 4, "index": 1},
        {"value": 5, "index": 1},
        {"value": 6, "index": 1},
    ]


class TestSequencePack(BaseDataAugmenterTest):
    augmenter = SequencePack(original_partition="TEST_PARTITION")
    input_features = {"values": Int, "trace_index": Int32}
    input_data = [
        {"values": 0, "trace_index": 0},
        {"values": 1, "trace_index": 0},
        {"values": 2, "trace_index": 0},
        {"values": 3, "trace_index": 1},
        {"values": 4, "trace_index": 1},
    ]
    expected_output_feature = Sequence[Int]
    expected_output_data = [[0, 1, 2], [3, 4]]
    expected_output_partition = "TEST_PARTITION"


class TestSequencePack(BaseDataAugmenterTest):
    augmenter = SequencePack(original_partition="TEST_PARTITION", original_length=3)
    input_features = {"values": Int, "trace_index": Int32}
    input_data = [
        {"values": 0, "trace_index": 0},
        {"values": 1, "trace_index": 0},
        {"values": 2, "trace_index": 0},
        {"values": 3, "trace_index": 1},
        {"values": 4, "trace_index": 1},
        {"values": 5, "trace_index": 1},
    ]
    expected_output_feature = Annotated[Sequence[Int], Len(3)]
    expected_output_data = [[0, 1, 2], [3, 4, 5]]
    expected_output_partition = "TEST_PARTITION"
