from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from hyped.core.features.dtypes import Int32Type
from hyped.core.graph import DataFlowGraph
from hyped.core.nodes.trace import TraceNode
from hyped.core.typing import IndexList, PartitionId, TraceIndexList


class TestTraceNode:
    @pytest.mark.parametrize(
        "path, traces, src_index, out_index, target_batch_size",
        [
            (
                (DataFlowGraph.Partition.CONST, DataFlowGraph.Partition.DEFAULT),
                {},
                [0],
                [0] * 16,
                16,
            ),
            (
                (DataFlowGraph.Partition.DEFAULT, "PARTITION_A"),
                {(DataFlowGraph.Partition.DEFAULT, "PARTITION_A"): []},
                [0, 1, 2, 3],
                [],
                0,
            ),
            (
                (DataFlowGraph.Partition.DEFAULT, "PARTITION_A"),
                {
                    (DataFlowGraph.Partition.DEFAULT, "PARTITION_A"): [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        3,
                    ]
                },
                [0, 1, 2, 3],
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                16,
            ),
            (
                (DataFlowGraph.Partition.DEFAULT, "PARTITION_A", "PARTITION_B"),
                {
                    (DataFlowGraph.Partition.DEFAULT, "PARTITION_A"): [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        3,
                        3,
                        3,
                        3,
                    ],
                    ("PARTITION_A", "PARTITION_B"): [0, 4, 8, 12],
                },
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                4,
            ),
        ],
    )
    def test_trace_through_partition_path(
        self,
        path: tuple[PartitionId],
        traces: dict[tuple[PartitionId, PartitionId], TraceIndexList],
        src_index: IndexList,
        out_index: IndexList,
        target_batch_size: int,
    ) -> None:
        mock_ctx = MagicMock(index=src_index, output_dtype=Int32Type)
        mock_values = pa.array(src_index, type=Int32Type.arrow_type)

        node = TraceNode(path=path)
        actual_out = node.trace_values_through_partition_path(
            mock_ctx, mock_values, target_batch_size, traces
        )

        assert actual_out.to_pylist() == out_index
