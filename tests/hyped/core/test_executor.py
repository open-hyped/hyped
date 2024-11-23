from typing import Hashable
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pyarrow as pa
import pytest

from hyped.core.executor import DataFlowExecutor
from hyped.core.features.reference import Reference
from hyped.core.graph import DataFlowGraph
from hyped.core.nodes.aggregator import BaseDataAggregator
from hyped.core.nodes.augmenter import BaseDataAugmenter
from hyped.core.nodes.collect import CollectNode
from hyped.core.nodes.processor import BaseDataProcessor

from .utils import build_graph_from_edge_list


def build_mock_node(node_type: DataFlowGraph.NodeType) -> MagicMock:
    """Builds a mock node based on the node type."""
    if node_type == DataFlowGraph.NodeType.SOURCE:
        return MagicMock()
    if node_type == DataFlowGraph.NodeType.CONST:
        return MagicMock(__class__=pa.Array)
    if node_type == DataFlowGraph.NodeType.COLLECT:
        return MagicMock(__class__=CollectNode, collect=MagicMock())
    if node_type == DataFlowGraph.NodeType.DATA_PROCESSOR:
        return MagicMock(__class__=BaseDataProcessor, run=AsyncMock())
    if node_type == DataFlowGraph.NodeType.DATA_AUGMENTER:
        return MagicMock(
            __class__=BaseDataAugmenter, run=AsyncMock(return_value=(MagicMock(), MagicMock()))
        )
    if node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
        return MagicMock(__class__=BaseDataAggregator)
    raise NotImplementedError(f"Unhandled node type: {node_type}")


@pytest.mark.asyncio
class TestDataFlowExecutor:
    @pytest.mark.parametrize(
        "edges, node_types, node, wait_for",
        [
            # Single data processor node
            (
                [(0, 1)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                1,
                [0],
            ),
            # Constant node
            (
                [(0, 1)],
                {
                    0: DataFlowGraph.NodeType.CONST,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                0,
                [],
            ),
            # Collect node with single input
            (
                [(0, 1)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.COLLECT,
                },
                1,
                [0],
            ),
            # Collect node with multiple inputs
            (
                [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    4: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                4,
                [1, 2, 3],
            ),
            # Augmenter node
            (
                [(0, 1)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTER,
                },
                1,
                [0],
            ),
            # Aggregator node
            (
                [(0, 1)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                },
                1,
                [0],
            ),
        ],
    )
    async def test_execute_node(
        self,
        edges: list[tuple[Hashable, Hashable]],
        node_types: dict[Hashable, DataFlowGraph.NodeType],
        node: Hashable,
        wait_for: list[Hashable],
    ) -> None:
        # Create mock inputs, index, and state
        mock_inputs = MagicMock()
        mock_index = MagicMock()
        mock_run_context = MagicMock()
        mock_run_context.output_type.arrow_type = MagicMock(__eq__=lambda *_: True)

        # Build the graph and set up mock nodes
        graph = build_graph_from_edge_list(edges, node_types)
        node_objs = {key: build_mock_node(node_type) for key, node_type in node_types.items()}
        nx.set_node_attributes(graph, node_objs, DataFlowGraph.NodeAttribute.NODE_OBJ)

        # Create mock state and manager
        state = MagicMock(
            wait_for=AsyncMock(),
            collect_inputs=MagicMock(return_value=(mock_inputs, mock_index)),
            register_partition_trace=MagicMock(),
        )
        manager = MagicMock(aggregate=AsyncMock())

        # Patch RunContext and initialize executor
        with patch("hyped.core.executor.RunContext", lambda *_, **__: mock_run_context):
            executor = DataFlowExecutor(graph, Reference(), manager)
            await executor.execute_node(node, state)

        # Verify awaited dependencies
        state.wait_for.assert_has_calls([((n,),) for n in wait_for])

        # Retrieve node object
        node_obj = node_objs[node]

        # Assertions based on node type
        if node_types[node] == DataFlowGraph.NodeType.CONST:
            state.capture_output.assert_called_once_with(node, node_obj)

        elif node_types[node] == DataFlowGraph.NodeType.COLLECT:
            node_obj.collect.assert_called_once_with(mock_run_context, mock_inputs)
            state.capture_output.assert_called_once_with(node, node_obj.collect())

        elif node_types[node] == DataFlowGraph.NodeType.DATA_PROCESSOR:
            node_obj.run.assert_called_once_with(mock_run_context, mock_inputs)
            state.capture_output.assert_called_once_with(node, await node_obj.run())

        elif node_types[node] == DataFlowGraph.NodeType.DATA_AUGMENTER:
            node_obj.run.assert_called_once_with(mock_run_context, mock_inputs)
            out, trace_index = await node_obj.run()
            state.register_partition_trace.assert_called_once_with(node, trace_index, mock_index)
            state.capture_output(node, out)

        elif node_types[node] == DataFlowGraph.NodeType.DATA_AGGREGATOR:
            manager.aggregate.assert_called_once_with(node_obj, mock_run_context, mock_inputs)

    @pytest.mark.asyncio
    async def test_execute(self) -> None:
        # Mock a collect function to simulate data collection
        collect = MagicMock()

        # Create mock objects to simulate batch, index, and rank inputs
        mock_batch = MagicMock()
        mock_index = MagicMock()
        mock_rank = MagicMock()

        # Mock the state object that will be used during execution
        mock_state = MagicMock(collect_value=MagicMock())

        with (
            # Patch the ExecutionState to use the mocked state object
            patch("hyped.core.executor.ExecutionState", lambda *_, **__: mock_state),
            # Patch execute_node to simulate asynchronous execution of nodes
            patch("hyped.core.executor.DataFlowExecutor.execute_node", AsyncMock()),
        ):
            # Build a sample graph with 4 nodes and define the executor
            graph = build_graph_from_edge_list([(0, 1), (1, 2), (2, 3)])
            executor = DataFlowExecutor(graph, collect=collect, aggregation_manager=None)

            # Run the execute method with mock inputs
            out = await executor.execute(mock_batch, mock_index, mock_rank)

            # Verify that execute_node was called for all nodes except the source node
            executor.execute_node.assert_has_calls(
                [
                    ((node_id, mock_state),)
                    for node_id in graph.nodes
                    if node_id != graph.src_node_id
                ]
            )

            # Ensure the collect_value method was called on the mock state with the collect function
            mock_state.collect_value.assert_called_once_with(collect)

            # Assert that the output matches the return value of collect_value
            assert out == mock_state.collect_value()
