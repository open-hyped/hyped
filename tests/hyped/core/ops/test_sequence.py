from unittest.mock import MagicMock, patch

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
    SequenceZip,
    zip_,
)
from hyped.core.testing.augmentor import BaseDataAugmentorTest
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


class TestSequenceUnpack(BaseDataAugmentorTest):
    augmentor = SequenceUnpack()
    input_features = {"seq": Sequence[Int]}
    input_data = [
        {"seq": [1, 2, 3]},
        {"seq": [4, 5, 6]},
    ]
    expected_output_feature = Int
    expected_output_data = [1, 2, 3, 4, 5, 6]


class TestSequenceUnpackWithIndex(BaseDataAugmentorTest):
    augmentor = SequenceUnpackWithIndex()
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


class TestSequencePack(BaseDataAugmentorTest):
    augmentor = SequencePack(original_partition="TEST_PARTITION")
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


class TestSequencePackFixedLength(BaseDataAugmentorTest):
    augmentor = SequencePack(original_partition="TEST_PARTITION", original_length=3)
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


class TestSequenceZip(BaseDataAugmentorTest):
    augmentor = SequenceZip()
    input_features = {"0": Sequence[Int32], "1": Sequence[Int32]}
    input_data = [
        {
            "0": [0, 1, 2, 3, 4],
            "1": [5, 6, 7, 8, 9],
        },
        {
            "0": [1, 2, 3],
            "1": [1, 2, 3],
        },
    ]
    expected_output_feature = Sequence[Annotated[Sequence[Int], Len(2)]]
    expected_output_data = [
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, 1], [2, 2], [3, 3]],
    ]


class TestSequenceZipMultipleInputs(BaseDataAugmentorTest):
    augmentor = SequenceZip()
    input_features = {"0": Sequence[Int32], "1": Sequence[Int32], "2": Sequence[Int32]}
    input_data = [
        {
            "0": [0, 1, 2, 3, 4],
            "1": [5, 6, 7, 8, 9],
            "2": [10, 11, 12, 13, 14],
        },
    ]
    expected_output_feature = Sequence[Annotated[Sequence[Int], Len(3)]]
    expected_output_data = [
        [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]],
    ]


class TestSequenceZipSorting(BaseDataAugmentorTest):
    augmentor = SequenceZip()
    input_features = {"1": Sequence[Int32], "0": Sequence[Int32]}
    input_data = [
        {
            "1": [5, 6, 7, 8, 9],
            "0": [0, 1, 2, 3, 4],
        },
        {
            "1": [1, 2, 3],
            "0": [1, 2, 3],
        },
    ]
    expected_output_feature = Sequence[Annotated[Sequence[Int], Len(2)]]
    expected_output_data = [
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, 1], [2, 2], [3, 3]],
    ]


class TestSequenceZipResolveLength(BaseDataAugmentorTest):
    augmentor = SequenceZip()
    input_features = {
        "0": Annotated[Sequence[Int32], Len(3)],
        "1": Annotated[Sequence[Int32], Len(3)],
    }
    input_data = [
        {
            "0": [0, 1, 2],
            "1": [5, 6, 7],
        },
        {
            "0": [1, 2, 3],
            "1": [1, 2, 3],
        },
    ]
    expected_output_feature = Annotated[
        Sequence[Annotated[Sequence[Int], Len(2, strict=True)]],
        Len(3, strict=True),
    ]
    expected_output_data = [
        [[0, 5], [1, 6], [2, 7]],
        [[1, 1], [2, 2], [3, 3]],
    ]


class TestSequenceZipLengthMismatch(BaseDataAugmentorTest):
    augmentor = SequenceZip()
    input_features = {
        "0": Annotated[Sequence[Int32], Len(3)],
        "1": Annotated[Sequence[Int32], Len(6)],
    }
    expected_verification_error = TypeError


def test_zip_():
    with patch("hyped.core.ops.sequence.SequenceZip") as SequenceZipMock:
        seq_1 = MagicMock()
        seq_2 = MagicMock()
        zip_(seq_1, seq_2)
        SequenceZipMock.return_value.call.assert_called_once_with(**{"0": seq_1, "1": seq_2})
