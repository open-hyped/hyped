from unittest.mock import MagicMock, patch

from hyped.core.graph import DataFlowGraph
from hyped.core.ops.sequence import (
    SequenceGetItem,
    SequenceGetItems,
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
    SequenceZipMapping,
    zip_,
)
from hyped.core.testing.augmentor import BaseDataAugmentorTest
from hyped.core.testing.processor import BaseDataProcessorTest
from hyped.typing import Annotated, Bool, Float64, Int, Int32, Len, Mapping, Sequence, String


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
    processor = SequenceGetItem()
    input_features = {"seq": Sequence[Int], "index": Int}
    input_data = [
        {"seq": [2, 1, 0], "index": 1},
        {"seq": [5, 3, 7], "index": 0},
    ]
    expected_output_feature = Int
    expected_output_data = [1, 5]


class TestSequenceGetItemNegative(BaseDataProcessorTest):
    processor = SequenceGetItem()
    input_features = {"seq": Sequence[Int], "index": Int}
    input_data = [
        {"seq": [2, 1, 0], "index": -2},
        {"seq": [5, 3, 7], "index": -1},
    ]
    expected_output_feature = Int
    expected_output_data = [1, 7]


class TestSequenceGetItems(BaseDataProcessorTest):
    processor = SequenceGetItems()
    input_features = {"seq": Sequence[Int], "index": Sequence[Int]}
    input_data = [
        {"seq": [2, 1, 0], "index": [2, 1, 0]},
        {"seq": [5, 3, 7], "index": [2, 0]},
    ]
    expected_output_feature = Sequence[Int]
    expected_output_data = [[0, 1, 2], [7, 5]]


class TestSequenceGetItemsFixedSizeList(BaseDataProcessorTest):
    processor = SequenceGetItems()
    input_features = {"seq": Annotated[Sequence[Int], Len(3)], "index": Sequence[Int]}
    input_data = [
        {"seq": [2, 1, 0], "index": [2, 1, 0]},
        {"seq": [5, 3, 7], "index": [2, 0]},
    ]
    expected_output_feature = Sequence[Int]
    expected_output_data = [[0, 1, 2], [7, 5]]


class TestSequenceGetItemsNegative(BaseDataProcessorTest):
    processor = SequenceGetItems()
    input_features = {"seq": Sequence[Int], "index": Sequence[Int]}
    input_data = [
        {"seq": [2, 1, 0], "index": [-2, 1, -1]},
        {"seq": [5, 3, 7], "index": [-1, 1]},
    ]
    expected_output_feature = Sequence[Int]
    expected_output_data = [[1, 1, 0], [7, 3]]


class TestSequenceGetItemsOutputLength(BaseDataProcessorTest):
    processor = SequenceGetItems()
    input_features = {"seq": Sequence[Int], "index": Annotated[Sequence[Int], Len(3)]}
    input_data = [
        {"seq": [2, 1, 0], "index": [-2, 1, -1]},
        {"seq": [5, 3, 7], "index": [-1, 1, 0]},
    ]
    expected_output_feature = Annotated[Sequence[Int], Len(3, strict=True)]
    expected_output_data = [[1, 1, 0], [7, 3, 5]]


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

    def test_output_partition(self) -> None:
        # call the node and capture the output partiton
        flow, feature = self.call_node(type(self).augmentor)
        partitionA = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # call it again and capture the second output partition
        flow, feature = self.call_node(type(self).augmentor)
        partitionB = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # for sequences of undefined length the output partition
        # is different
        assert partitionA != partitionB


class TestFixedLengthSequenceUnpack(BaseDataAugmentorTest):
    augmentor = SequenceUnpack()
    input_features = {"seq": Annotated[Sequence[Int], Len(3)]}

    def test_output_partition(self) -> None:
        # call the node and capture the output partiton
        flow, feature = self.call_node(type(self).augmentor)
        partitionA = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # call it again and capture the second output partition
        flow, feature = self.call_node(type(self).augmentor)
        partitionB = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # for sequences of undefined length the output partition
        # is different
        assert partitionA == partitionB


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

    def test_output_partition(self) -> None:
        # call the node and capture the output partiton
        flow, feature = self.call_node(type(self).augmentor)
        partitionA = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # call it again and capture the second output partition
        flow, feature = self.call_node(type(self).augmentor)
        partitionB = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # for sequences of undefined length the output partition
        # is different
        assert partitionA != partitionB


class TestFixedLengthSequenceUnpackWithIndex(BaseDataAugmentorTest):
    augmentor = SequenceUnpackWithIndex()
    input_features = {"seq": Annotated[Sequence[Int], Len(3)]}

    def test_output_partition(self) -> None:
        # call the node and capture the output partiton
        flow, feature = self.call_node(type(self).augmentor)
        partitionA = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # call it again and capture the second output partition
        flow, feature = self.call_node(type(self).augmentor)
        partitionB = flow._graph.nodes[feature.ref._node_id][
            DataFlowGraph.NodeAttribute.OUT_PARTITION
        ]
        # for sequences of undefined length the output partition
        # is different
        assert partitionA == partitionB


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


class TestSequenceZip(BaseDataProcessorTest):
    processor = SequenceZip()
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


class TestSequenceZipMultipleInputs(BaseDataProcessorTest):
    processor = SequenceZip()
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


class TestSequenceZipSorting(BaseDataProcessorTest):
    processor = SequenceZip()
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


class TestSequenceZipResolveLength(BaseDataProcessorTest):
    processor = SequenceZip()
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


class TestSequenceZipLengthMismatch(BaseDataProcessorTest):
    processor = SequenceZip()
    input_features = {
        "0": Annotated[Sequence[Int32], Len(3)],
        "1": Annotated[Sequence[Int32], Len(6)],
    }
    expected_verification_error = TypeError


class ExpectedSequenceMapping(Mapping):
    A: String
    B: Int32


class TestSequenceZipMapping(BaseDataProcessorTest):
    processor = SequenceZipMapping()
    input_features = {
        "A": Sequence[String],
        "B": Sequence[Int32],
    }
    input_data = [
        {
            "A": ["A", "B", "C"],
            "B": [1, 2, 3],
        },
        {
            "A": ["D", "E"],
            "B": [4, 5],
        },
    ]
    expected_output_feature = Sequence[ExpectedSequenceMapping]
    expected_output_data = [
        [
            {"A": "A", "B": 1},
            {"A": "B", "B": 2},
            {"A": "C", "B": 3},
        ],
        [
            {"A": "D", "B": 4},
            {"A": "E", "B": 5},
        ],
    ]


class ExpectedSequenceMappingMultipleInputs(Mapping):
    A: String
    B: Int32
    C: Float64


class TestSequenceZipMappingMultipleInputs(BaseDataProcessorTest):
    processor = SequenceZipMapping()
    input_features = {
        "A": Sequence[String],
        "B": Sequence[Int32],
        "C": Sequence[Float64],
    }
    input_data = [
        {
            "A": ["A", "B", "C"],
            "B": [1, 2, 3],
            "C": [0.0, 2.2, 3.3],
        },
    ]
    expected_output_feature = Sequence[ExpectedSequenceMappingMultipleInputs]
    expected_output_data = [
        [
            {"A": "A", "B": 1, "C": 0.0},
            {"A": "B", "B": 2, "C": 2.2},
            {"A": "C", "B": 3, "C": 3.3},
        ],
    ]


class TestSequenceZipMappingResolveLength(BaseDataProcessorTest):
    processor = SequenceZipMapping()
    input_features = {
        "A": Annotated[Sequence[String], Len(3)],
        "B": Annotated[Sequence[Int32], Len(3)],
    }
    input_data = [
        {
            "A": ["A", "B", "C"],
            "B": [1, 2, 3],
        },
        {
            "A": ["D", "E", "F"],
            "B": [4, 5, 6],
        },
    ]
    expected_output_feature = Annotated[Sequence[ExpectedSequenceMapping], Len(3, strict=True)]
    expected_output_data = [
        [
            {"A": "A", "B": 1},
            {"A": "B", "B": 2},
            {"A": "C", "B": 3},
        ],
        [
            {"A": "D", "B": 4},
            {"A": "E", "B": 5},
            {"A": "F", "B": 6},
        ],
    ]


class TestSequenceZipMappingLengthMismatch(BaseDataProcessorTest):
    processor = SequenceZipMapping()
    input_features = {
        "A": Annotated[Sequence[String], Len(3)],
        "B": Annotated[Sequence[Int32], Len(2)],
    }
    expected_verification_error = TypeError


def test_zip_():
    with patch("hyped.core.ops.sequence.SequenceZip") as SequenceZipMock:
        seq_1 = MagicMock()
        seq_2 = MagicMock()
        zip_(seq_1, seq_2)
        SequenceZipMock.return_value.call.assert_called_once_with(**{"0": seq_1, "1": seq_2})


def test_zip_kwargs():
    with patch("hyped.core.ops.sequence.SequenceZipMapping") as SequenceZipMock:
        seq_1 = MagicMock()
        seq_2 = MagicMock()
        zip_(A=seq_1, B=seq_2)
        SequenceZipMock.return_value.call.assert_called_once_with(**{"A": seq_1, "B": seq_2})
