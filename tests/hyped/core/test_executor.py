import asyncio
from types import MappingProxyType
from typing import Generator, Hashable
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from hyped.core.executor import DataFlowExecutor, ExecutionState, LazyDataFlowExecutor
from hyped.core.features.dtypes import BoolType, DType, MappingType
from hyped.core.features.reference import ForwardReference
from hyped.core.graph import DataFlowGraph
from hyped.core.nodes.aggregator import BaseDataAggregator
from hyped.core.nodes.augmentor import BaseDataAugmentor
from hyped.core.nodes.base import BaseNode, BaseNodeConfig
from hyped.core.nodes.collect import CollectNode
from hyped.core.nodes.const import ConstNode
from hyped.core.nodes.processor import BaseDataProcessor
from hyped.core.typing import PartitionId

from .utils import NumpyArrayMatcher, build_graph


def build_mock_node(node_type: DataFlowGraph.NodeType) -> MagicMock:
    """Builds a mock node based on the node type."""
    if node_type == DataFlowGraph.NodeType.SOURCE:
        return MagicMock()
    if node_type == DataFlowGraph.NodeType.CONST:
        return MagicMock(spec=ConstNode)
    if node_type == DataFlowGraph.NodeType.CAST:
        return MagicMock(spec=DType)
    if node_type == DataFlowGraph.NodeType.COLLECT:
        return MagicMock(spec=CollectNode, collect=MagicMock())
    if node_type == DataFlowGraph.NodeType.DATA_PROCESSOR:
        return MagicMock(spec=BaseDataProcessor, run=AsyncMock())
    if node_type == DataFlowGraph.NodeType.DATA_AUGMENTOR:
        return MagicMock(
            spec=BaseDataAugmentor, run=AsyncMock(return_value=(MagicMock(), MagicMock()))
        )
    if node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
        return MagicMock(spec=BaseDataAggregator)
    raise NotImplementedError(f"Unhandled node type: {node_type}")


class TestExecutionState:
    @pytest.mark.asyncio
    async def test_wait_for(self) -> None:
        # create a simple graph with one processor node
        graph = build_graph(
            edges=[(0, 1)],
            node_types={0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
        )

        # create the execution state
        state = ExecutionState(
            graph, graph.build_partition_graph(), MagicMock(), MagicMock(), MagicMock()
        )

        # create a tasks waiting for the node to get ready
        wait_task = asyncio.create_task(state.wait_for(1))

        # make sure the task doesn't complete
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=0.1)

        async def simulate_ready():
            # simulate the node ready attribute to be set
            await asyncio.sleep(0.1)
            state.ready[1].set()

        # create a tasks to mark the node as ready and wait for both tasks to finish
        ready_task = asyncio.create_task(simulate_ready())
        await asyncio.gather(wait_task, ready_task)

        # ensure the wait for operation completes
        assert wait_task.done()

    @pytest.mark.parametrize(
        "edges, node_types, partitions, source_partition, target_partition, trace_index",
        [
            # Stay in default partition
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                {},
                DataFlowGraph.Partition.DEFAULT,
                DataFlowGraph.Partition.DEFAULT,
                [0, 1, 2, 3, 4],
            ),
            # Simple filter augmentor
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                {
                    1: [0, 1, 3, 4],
                },
                DataFlowGraph.Partition.DEFAULT,
                1,
                [0, 1, 3, 4],
            ),
            # Simple augmentation
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                {
                    1: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                },
                DataFlowGraph.Partition.DEFAULT,
                1,
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            ),
            # Chaining filters
            (
                [(0, 1, "x"), (1, 2, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                {1: [1, 2, 3], 2: [1, 2]},
                DataFlowGraph.Partition.DEFAULT,
                2,
                [2, 3],
            ),
            # Chaining augmentors
            (
                [(0, 1, "x"), (1, 2, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                {1: [1, 1, 1, 2, 2], 2: [2, 3, 4, 4]},
                DataFlowGraph.Partition.DEFAULT,
                2,
                [1, 2, 2, 2],
            ),
        ],
    )
    def test_trace_through_partition_path(
        self,
        edges: list[tuple[Hashable, Hashable]],
        node_types: dict[Hashable, DataFlowGraph.NodeType],
        partitions: dict[PartitionId, np.ndarray],
        source_partition: Hashable,
        target_partition: Hashable,
        trace_index: np.ndarray,
    ) -> None:
        index = [0, 1, 2, 3, 4]
        trace_index = np.asarray(trace_index)

        # mock augmentor nodes
        mock_augmentor_nodes = {
            i: MagicMock(
                __spec__=BaseDataAugmentor, infer_output_partition=MagicMock(return_value=i)
            )
            for i, node_type in node_types.items()
            if node_type == DataFlowGraph.NodeType.DATA_AUGMENTOR
        }

        # build the data flow graph
        graph = build_graph(edges, node_types, node_objects=mock_augmentor_nodes)
        state = ExecutionState(
            graph, graph.build_partition_graph(), MagicMock(), index, MagicMock()
        )

        # register partition traces
        for p in nx.topological_sort(graph.build_partition_graph()):
            if p in partitions:
                state.register_partition_trace(p, partitions[p])

        # create a mock value to be traced
        # must have the same length as the index
        mock = MagicMock(__len__=lambda _: len(index))

        # trace the mock value trough the partition path
        traced_mock = state.trace_through_partition_path(
            [mock], source_partition, target_partition
        )[0]

        mock.take.assert_called_once_with(NumpyArrayMatcher(trace_index))
        assert traced_mock == mock.take.return_value

    @patch("hyped.core.executor.pa.array")
    @patch("hyped.core.executor.pa.chunked_array")
    @patch("hyped.core.executor.ExecutionState.trace_through_partition_path")
    def test_collect_inputs(
        self,
        mock_trace_trough_partition_graph: MagicMock,
        mock_arrow_chunked_array: MagicMock,
        mock_arrow_array: MagicMock,
    ) -> None:
        mock_trace_trough_partition_graph.return_value = (MagicMock(),)

        # create a mock graph including a constant and a data augmentor
        # to introduce another partition
        graph = build_graph(
            [(0, 1, "0"), (0, 2, "0"), (3, 4, "3"), (1, 4, "1"), (2, 4, "2")],
            {
                0: DataFlowGraph.NodeType.SOURCE,
                1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                3: DataFlowGraph.NodeType.CONST,
                4: DataFlowGraph.NodeType.DATA_PROCESSOR,
            },
            {
                2: MagicMock(
                    __spec__=BaseDataAugmentor, infer_output_partition=MagicMock(return_value=2)
                )
            },
        )

        # create a mock index of length 5
        mock_index = MagicMock(__len__=lambda _: 5)
        # create execution state
        state = ExecutionState(
            graph, graph.build_partition_graph(), MagicMock(), mock_index, MagicMock()
        )
        state.index[2] = mock_index

        mock_outputs = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        # capture required outputs
        state.capture_output(1, mock_outputs[1])
        state.capture_output(2, mock_outputs[2])
        state.capture_output(3, mock_outputs[3])

        # collect input for node
        inputs, index = state.collect_inputs(4)

        # the inputs from the data processor must be passed through
        # the partition graph
        mock_trace_trough_partition_graph.assert_called_once_with(
            [mock_outputs[1]], src=DataFlowGraph.Partition.DEFAULT, tgt=2
        )
        assert inputs["1"] == mock_trace_trough_partition_graph.return_value[0]

        # the output of the data augmentor introducing the partition
        # is not traced through the graph
        assert inputs["2"] == mock_outputs[2]

        # make sure the constant input is a chunked array
        mock_arrow_chunked_array.assert_called_once_with([mock_outputs[3]] * 5)
        assert inputs["3"] == mock_arrow_chunked_array.return_value


class TestDataFlowExecutor:
    def test_init(self) -> None:
        # create a simple data flow graph without data aggregators
        graph = build_graph(
            edges=[(0, 1, "x")],
            node_types={0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
        )
        # initialize an executor for the graph without an aggregation manager
        DataFlowExecutor(graph, ForwardReference(), None)

        # create a simple data flow graph with a data aggregator
        graph = build_graph(
            edges=[(0, 1, "x")],
            node_types={
                0: DataFlowGraph.NodeType.SOURCE,
                1: DataFlowGraph.NodeType.DATA_AGGREGATOR,
            },
        )

        with pytest.raises(RuntimeError):
            # cannot initialize an executor without an aggregation manager
            DataFlowExecutor(graph, ForwardReference(), None)

        # initialize the executor with a mock aggregation manager
        DataFlowExecutor(graph, ForwardReference(), MagicMock())

    @pytest.mark.parametrize(
        "edges, node_types, node, wait_for",
        [
            # Single data processor node
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                1,
                [0],
            ),
            # Constant node
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.CONST,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                0,
                [],
            ),
            # Cast node
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.CAST,
                },
                1,
                [0],
            ),
            # Collect node with single input
            (
                [(0, 1, "value")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.COLLECT,
                },
                1,
                [0],
            ),
            # Collect node with multiple inputs
            (
                [(0, 1, "x"), (0, 2, "x"), (0, 3, "x"), (1, 4, "a"), (2, 4, "b"), (3, 4, "c")],
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
            # Augmentor node
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                1,
                [0],
            ),
            # Aggregator node
            (
                [(0, 1, "x")],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                },
                1,
                [0],
            ),
        ],
    )
    @pytest.mark.asyncio
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
        graph = build_graph(edges, node_types)
        node_objs = {key: build_mock_node(node_type) for key, node_type in node_types.items()}
        nx.set_node_attributes(graph, node_objs, DataFlowGraph.NodeAttribute.NODE_OBJ)

        # Create mock state and manager
        state = MagicMock(
            wait_for=AsyncMock(),
            collect_inputs=MagicMock(return_value=(mock_inputs, mock_index)),
            register_partition_trace=MagicMock(),
        )
        manager = MagicMock(aggregate=AsyncMock())

        mock_pyarrow_cast = MagicMock()
        # Patch RunContext and initialize executor
        with patch("hyped.core.executor.RunContext", lambda *_, **__: mock_run_context), patch(
            "hyped.core.executor.pc.cast", mock_pyarrow_cast
        ):
            executor = DataFlowExecutor(graph, ForwardReference(), manager)
            executor.session = MagicMock()
            await executor.execute_node(node, state)

        # Verify awaited dependencies
        state.wait_for.assert_has_calls([((n,),) for n in wait_for])

        # Retrieve node object
        node_obj = node_objs[node]

        # Assertions based on node type
        if node_types[node] == DataFlowGraph.NodeType.CONST:
            state.capture_output.assert_called_once_with(node, node_obj.config.value)

        elif node_types[node] == DataFlowGraph.NodeType.CAST:
            mock_value = mock_inputs.__getitem__.return_value
            mock_pyarrow_cast.assert_called_once_with(mock_value, ANY)
            state.capture_output.assert_called_once_with(node, mock_pyarrow_cast.return_value)

        elif node_types[node] == DataFlowGraph.NodeType.COLLECT:
            node_obj.collect.assert_called_once_with(mock_run_context, mock_inputs)
            state.capture_output.assert_called_once_with(node, node_obj.collect.return_value)

        elif node_types[node] == DataFlowGraph.NodeType.DATA_PROCESSOR:
            node_obj.run.assert_called_once_with(mock_run_context, mock_inputs)
            state.capture_output.assert_called_once_with(node, node_obj.run.return_value)

        elif node_types[node] == DataFlowGraph.NodeType.DATA_AUGMENTOR:
            node_obj.run.assert_called_once_with(mock_run_context, mock_inputs)
            out, trace_index = await node_obj.run()
            state.register_partition_trace.assert_called_once_with(node, trace_index)
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
            graph = build_graph([(0, 1, "x"), (1, 2, "x"), (2, 3, "x")])
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
            assert out == mock_state.collect_value.return_value

    def test_run(self) -> None:
        # create an executor instance
        executor = DataFlowExecutor(MagicMock(), MagicMock(), None)
        executor.execute = AsyncMock()
        executor.init_run_session = MagicMock(side_effect=executor.init_run_session)

        with (
            patch("hyped.core.executor.pa.table") as mock_table,
            patch("hyped.core.executor.get_worker_info") as mock_get_worker_info,
        ):
            # create mock inputs to be processed
            mock_batch = MagicMock()
            mock_index = MagicMock()

            # run the executor
            out = executor.run(mock_batch, mock_index, None)
            assert out == mock_table.return_value

            # make sure the session was initialized
            executor.init_run_session.assert_called_once()
            # make sure execute was called correctly
            executor.execute.assert_called_once_with(
                mock_batch.to_struct_array.return_value,
                index=mock_index,
                rank=mock_get_worker_info.return_value.rank,
            )

    @pytest.mark.asyncio
    async def test_state_stored_in_session(self) -> None:
        mock_variable = MagicMock()

        class MockConfig(BaseNodeConfig):
            ...

        class MockNode(BaseNode[MockConfig]):
            @property
            def signature(self):
                ...

            def initialize(self, ctx):
                self.variable = mock_variable

            async def run(self, ctx, inputs):
                # make sure the variable is present
                # when executing the node
                assert hasattr(self, "variable")
                assert self.variable == mock_variable
                return MagicMock(type=BoolType.arrow_type)

        # build a mock data flow graph
        mock_graph = nx.MultiDiGraph()
        mock_graph.add_node(
            "node_id",
            **{
                DataFlowGraph.NodeAttribute.NODE_OBJ: MockNode(),
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE: MagicMock(),
                DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE: BoolType,
                DataFlowGraph.NodeAttribute.PARTITION: DataFlowGraph.Partition.DEFAULT,
                DataFlowGraph.NodeAttribute.DEPTH: 0,
            },
        )
        mock_graph.build_partition_graph = MagicMock()

        # initialize the session
        executor = DataFlowExecutor(mock_graph, MagicMock(), None)
        executor.init_run_session(0)

        # make sure initialize has no effect on the node state
        assert not hasattr(executor, "variable")
        # make sure the initialized node state is stored in the session
        state = executor.session.get_context("node_id")
        assert state is not None
        assert state["__dict__"]["variable"] == mock_variable

        # make sure the state is set when executing the node
        mock_state = MagicMock(collect_inputs=MagicMock(return_value=(MagicMock(), MagicMock())))
        await executor.execute_node("node_id", mock_state)

        # reset the session
        executor.reset_run_session()
        assert executor.session is None


class TestLazyDataFlowExecutor:
    @pytest.fixture(scope="function")
    def mock_inputs(self) -> dict[str, MagicMock]:
        return {"x": MagicMock()}

    @pytest.fixture
    def mock_arrow_table(self) -> Generator[MagicMock, None, None]:
        with patch("hyped.core.executor.pa.table") as mock:
            yield mock

    @pytest.fixture
    def lazy_executor(
        self, mock_inputs: dict[str, MagicMock], mock_arrow_table: MagicMock
    ) -> LazyDataFlowExecutor:
        # create a mock graph with the required functionality
        mock_graph = MagicMock()
        mock_graph.get_dtype_from_reference.return_value = MappingType.construct({"y": BoolType})

        # create the lazy executor instance
        return LazyDataFlowExecutor(mock_graph, MagicMock(), MappingProxyType(mock_inputs))

    def test_get_item(
        self,
        lazy_executor: LazyDataFlowExecutor,
        mock_inputs: dict[str, MagicMock],
        mock_arrow_table: MagicMock,
    ) -> None:
        with pytest.raises(KeyError):
            # request invalid key not contained in output mapping type
            lazy_executor["invalid_key"]

        with patch("hyped.core.executor.DataFlowExecutor.execute", AsyncMock()) as mock_execute:
            # request valid key
            y = lazy_executor["y"]

            # make sure execute was called correctly
            mock_execute.assert_called_once_with(
                mock_arrow_table.return_value.to_struct_array.return_value, index=[0], rank=0
            )
            # expected output snapshot
            expected_out_snapshot = (
                mock_execute.return_value.__getitem__.return_value.as_py.return_value
            )
            # check output
            expected_out_snapshot.__getitem__.assert_called_once_with("y")
            assert y == expected_out_snapshot.__getitem__.return_value

            # reset execute mock function
            mock_execute.reset_mock()
            # request valid key again without input changing
            y = lazy_executor["y"]
            # make sure the execute function is not called again
            mock_execute.assert_not_called()

            # reset execute mock function
            mock_execute.reset_mock()
            # change the input
            mock_inputs["x"] = MagicMock()
            # request valid key again without input changing
            y = lazy_executor["y"]
            # make sure the execute function is not called again
            mock_execute.assert_called_once()

    def test_mapping(self, lazy_executor: LazyDataFlowExecutor) -> None:
        assert len(lazy_executor) == 1
        # test key access
        assert set(list(lazy_executor.keys())) == {"y"}
        assert set(list(iter(lazy_executor))) == {"y"}

        with patch("hyped.core.executor.DataFlowExecutor.execute", AsyncMock()):
            # convert to dictionary and check keys again
            mapping = dict(lazy_executor)
            assert set(list(mapping.keys())) == {"y"}

        # run string conversion
        str(lazy_executor)
