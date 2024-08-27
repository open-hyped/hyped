import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import networkx as nx
import numpy as np
import pytest
from datasets import Features, Value

from hyped.data.flow.core.executor import (
    DataFlowExecutor,
    ExecutionState,
    _gather,
)
from hyped.data.flow.core.graph import DataFlowGraph
from hyped.data.flow.core.nodes.base import IOContext


def test_gather():
    assert _gather([10, 20, 30, 40, 50], [0, 2, 4]) == [10, 30, 50]
    assert _gather([10, 20, 30, 40, 50], []) == []
    assert _gather([10, 20, 30, 40, 50], [4, 2, 0]) == [50, 30, 10]
    assert _gather([10, 20, 30, 40, 50], [1, 1, 1]) == [20, 20, 20]
    assert _gather([10], [0]) == [10]


class TestExecutionState:
    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        mock_in_features = Features({"a": Value("int64"), "b": Value("int64")})
        mock_out_features = Features({"y": Value("int64")})

        G = nx.MultiDiGraph()
        # add the source node to the graph
        G.add_node(
            "SOURCE_NODE",
            **{
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.SOURCE,
                DataFlowGraph.NodeAttribute.NODE_OBJ: None,  # no node object for source nodes
                DataFlowGraph.NodeAttribute.PARTITION: DataFlowGraph.PredefinedPartition.DEFAULT,
                DataFlowGraph.NodeAttribute.IN_FEATURES: None,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: Features(
                    {"x": Value("int64")}
                ),
            },
        )

        # add a constant value node to the graph
        # not connected by default
        G.add_node(
            "CONST_NODE",
            **{
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.CONST,
                DataFlowGraph.NodeAttribute.NODE_OBJ: MagicMock(),
                DataFlowGraph.NodeAttribute.PARTITION: DataFlowGraph.PredefinedPartition.CONST,
                DataFlowGraph.NodeAttribute.IN_FEATURES: None,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: Features(
                    {"value": Value("int64")}
                ),
            },
        )

        # add a mock processor
        G.add_node(
            "PROCESSOR_NODE",
            **{
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeAttribute.NODE_OBJ: MagicMock(),
                DataFlowGraph.NodeAttribute.PARTITION: DataFlowGraph.PredefinedPartition.DEFAULT.value,
                DataFlowGraph.NodeAttribute.IN_FEATURES: mock_in_features,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: mock_out_features,
            },
        )
        for name in ("a", "b"):
            G.add_edge(
                "SOURCE_NODE",
                "PROCESSOR_NODE",
                key=name,
                **{
                    DataFlowGraph.EdgeAttribute.NAME: name,
                    DataFlowGraph.EdgeAttribute.KEY: MagicMock(),
                },
            )

        # add a mock augmenter
        G.add_node(
            "AUGMENTER_NODE",
            **{
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_AUGMENTER,
                DataFlowGraph.NodeAttribute.NODE_OBJ: MagicMock(),
                DataFlowGraph.NodeAttribute.PARTITION: DataFlowGraph.PredefinedPartition.DEFAULT.value,
                DataFlowGraph.NodeAttribute.IN_FEATURES: mock_in_features,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: mock_out_features,
            },
        )
        for name in ("a", "b"):
            G.add_edge(
                "PROCESSOR_NODE",
                "AUGMENTER_NODE",
                key=name,
                **{
                    DataFlowGraph.EdgeAttribute.NAME: name,
                    DataFlowGraph.EdgeAttribute.KEY: MagicMock(),
                },
            )

        # add a mock processor to the augmenter partition
        G.add_node(
            "PROCESSOR_NODE_IN_AUGMENTER_PARTITION",
            **{
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeAttribute.NODE_OBJ: MagicMock(),
                DataFlowGraph.NodeAttribute.PARTITION: "AUGMENTER_NODE",
                DataFlowGraph.NodeAttribute.IN_FEATURES: mock_in_features,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: mock_out_features,
            },
        )
        for name in ("a", "b"):
            G.add_edge(
                "AUGMENTER_NODE",
                "PROCESSOR_NODE_IN_AUGMENTER_PARTITION",
                key=name,
                **{
                    DataFlowGraph.EdgeAttribute.NAME: name,
                    DataFlowGraph.EdgeAttribute.KEY: MagicMock(),
                },
            )

        # add a mock augmenter
        G.add_node(
            "AUGMENTER_NODE_2",
            **{
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_AUGMENTER,
                DataFlowGraph.NodeAttribute.NODE_OBJ: MagicMock(),
                DataFlowGraph.NodeAttribute.PARTITION: "AUGMENTER_NODE",
                DataFlowGraph.NodeAttribute.IN_FEATURES: mock_in_features,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: mock_out_features,
            },
        )
        for name in ("a", "b"):
            G.add_edge(
                "PROCESSOR_NODE_IN_AUGMENTER_PARTITION",
                "AUGMENTER_NODE_2",
                key=name,
                **{
                    DataFlowGraph.EdgeAttribute.NAME: name,
                    DataFlowGraph.EdgeAttribute.KEY: MagicMock(),
                },
            )

        # add a mock processor to the augmenter partition
        G.add_node(
            "PROCESSOR_NODE_IN_AUGMENTER_PARTITION_2",
            **{
                DataFlowGraph.NodeAttribute.NODE_TYPE: DataFlowGraph.NodeType.DATA_PROCESSOR,
                DataFlowGraph.NodeAttribute.NODE_OBJ: MagicMock(),
                DataFlowGraph.NodeAttribute.PARTITION: "AUGMENTER_NODE_2",
                DataFlowGraph.NodeAttribute.IN_FEATURES: mock_in_features,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: mock_out_features,
            },
        )
        for name in ("a", "b"):
            G.add_edge(
                "AUGMENTER_NODE_2",
                "PROCESSOR_NODE_IN_AUGMENTER_PARTITION_2",
                key=name,
                **{
                    DataFlowGraph.EdgeAttribute.NAME: name,
                    DataFlowGraph.EdgeAttribute.KEY: MagicMock(),
                },
            )

        mock_G = MagicMock(
            wraps=G, nodes=G.nodes, edges=G.edges, src_node_id="SOURCE_NODE"
        )
        mock_G.src_node_id = "SOURCE_NODE"
        mock_G.get_node_output_partition = {
            "CONST_NODE": DataFlowGraph.PredefinedPartition.CONST,
            "SOURCE_NODE": DataFlowGraph.PredefinedPartition.DEFAULT,
            "PROCESSOR_NODE": DataFlowGraph.PredefinedPartition.DEFAULT,
            "AUGMENTER_NODE": "AUGMENTER_NODE",
            "PROCESSOR_NODE_IN_AUGMENTER_PARTITION": "AUGMENTER_NODE",
            "AUGMENTER_NODE_2": "AUGMENTER_NODE_2",
            "PROCESSOR_NODE_IN_AUGMENTER_PARTITION_2": "AUGMENTER_NODE_2",
        }.get

        return mock_G

    @pytest.fixture
    def mock_p_graph(self) -> nx.DiGraph:
        G = nx.path_graph(
            [
                DataFlowGraph.PredefinedPartition.DEFAULT.value,
                "AUGMENTER_NODE",
                "AUGMENTER_NODE_2",
            ],
            create_using=nx.DiGraph,
        )
        G.add_node(DataFlowGraph.PredefinedPartition.CONST)
        return G

    @pytest.fixture
    def mock_index(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_batch(self, mock_index) -> MagicMock:
        mock_batch = MagicMock()
        # items in batch must have the same length as the index
        mock_batch.__getitem__().__len__ = MagicMock(
            return_value=mock_index.__len__()
        )
        return mock_batch

    @pytest.fixture
    def mock_state(
        self,
        mock_graph: MagicMock,
        mock_p_graph: nx.DiGraph,
        mock_batch: MagicMock,
        mock_index: MagicMock,
    ) -> ExecutionState:
        return ExecutionState(
            mock_graph, mock_p_graph, mock_batch, mock_index, rank=0
        )

    def test_initial_state(self, mock_state: ExecutionState):
        assert set(mock_state.outputs.keys()) == {"SOURCE_NODE"}
        assert set(mock_state.ready.keys()) == {
            "CONST_NODE",
            "PROCESSOR_NODE",
            "AUGMENTER_NODE",
            "PROCESSOR_NODE_IN_AUGMENTER_PARTITION",
            "AUGMENTER_NODE_2",
            "PROCESSOR_NODE_IN_AUGMENTER_PARTITION_2",
        }

    @pytest.mark.asyncio
    async def test_wait_for(self, mock_state: ExecutionState):
        # make sure the node is not ready yet
        assert not mock_state.ready["PROCESSOR_NODE"].is_set()

        # This coroutine should block until the event is set
        async def wait():
            await mock_state.wait_for("PROCESSOR_NODE")
            return True

        # schedule coroutine
        task = asyncio.create_task(wait())
        await asyncio.sleep(0.1)  # Ensure the task is waiting

        # set ready event
        mock_state.ready["PROCESSOR_NODE"].set()
        assert (await task) is True

    def test_register_partition_trace(self, mock_state: ExecutionState):
        mock_return_index = MagicMock()
        with patch(
            "hyped.data.flow.core.executor.ExecutionState.trace_through_partition_path",
            MagicMock(return_value=[mock_return_index]),
        ) as mock:
            # create mock index and trace index values
            index = list(range(10, 20))
            trace_index = list(range(10))

            # register the partition trace
            mock_state.register_partition_trace(
                "AUGMENTER_NODE", trace_index, index=index
            )

            # check the stored trace index for the transition between the two partitions
            assert (
                mock_state.traces[
                    (
                        DataFlowGraph.PredefinedPartition.DEFAULT,
                        "AUGMENTER_NODE",
                    )
                ]
                == np.asarray(trace_index)
            ).all()

            # make sure the index is traced through the path
            mock.assert_called_once_with(
                [index],
                src=DataFlowGraph.PredefinedPartition.DEFAULT,
                tgt="AUGMENTER_NODE",
            )
            # make sure the stored index is the result of the call
            assert mock_state.index["AUGMENTER_NODE"] is mock_return_index

    def test_trace_through_partition_path(self, mock_state: ExecutionState):
        # create mock objects
        mock_value = MagicMock()
        mock_index = MagicMock()
        mock_trace_index_1 = MagicMock()
        mock_trace_index_2 = MagicMock()
        # the function checks that the length of the provided
        # values matches the length of the index
        mock_value.__len__ = mock_index.__len__ = MagicMock()

        # insert mock obejcts into mock execution state
        mock_state.traces = {
            (
                DataFlowGraph.PredefinedPartition.DEFAULT.value,
                "AUGMENTER_NODE",
            ): mock_trace_index_1,
            ("AUGMENTER_NODE", "AUGMENTER_NODE_2"): mock_trace_index_2,
        }
        mock_state.index = {
            DataFlowGraph.PredefinedPartition.DEFAULT: mock_index,
            "AUGMENTER_NODE": mock_index,
        }

        # patch the conversion from list to numpy array
        with patch(
            "hyped.data.flow.core.executor.np.arange", lambda _: mock_index
        ), patch("hyped.data.flow.core.executor._gather", lambda v, p: (v, p)):
            # apply the function to test
            mock_output = mock_state.trace_through_partition_path(
                [mock_value],
                src=DataFlowGraph.PredefinedPartition.DEFAULT.value,
                tgt="AUGMENTER_NODE",
            )

            assert len(mock_output) == 1
            src_values, applied_index = mock_output[0]
            assert src_values == mock_value
            assert applied_index == mock_index[mock_trace_index_1]

            # apply the function to test
            mock_output = mock_state.trace_through_partition_path(
                [mock_value], src="AUGMENTER_NODE", tgt="AUGMENTER_NODE_2"
            )

            assert len(mock_output) == 1
            src_values, applied_index = mock_output[0]
            assert src_values == mock_value
            assert applied_index == mock_index[mock_trace_index_2]

            # apply the function to test
            mock_output = mock_state.trace_through_partition_path(
                [mock_value],
                src=DataFlowGraph.PredefinedPartition.DEFAULT.value,
                tgt="AUGMENTER_NODE_2",
            )

            assert len(mock_output) == 1
            src_values, applied_index = mock_output[0]
            assert src_values == mock_value
            assert (
                applied_index
                == mock_index[mock_trace_index_2][mock_trace_index_1]
            )

    def test_collect_value(self, mock_state: ExecutionState):
        # create mock objects
        mock_node_id = MagicMock()
        mock_node_output = MagicMock()
        # create mock feature reference
        mock_feature_ref = MagicMock()
        mock_feature_ref.feature_ = Features()
        mock_feature_ref.node_id_ = mock_node_id

        # add mock output to state
        mock_state.outputs[mock_node_id] = mock_node_output

        with patch(
            "hyped.data.flow.core.executor.list_of_dicts_to_dict_of_lists"
        ) as mock_convert:
            # collect mock output and check return value
            out = mock_state.collect_value(mock_feature_ref)
            assert out == mock_convert(
                mock_feature_ref.key_.index_batch(mock_node_output),
                keys=mock_feature_ref.feature_.keys(),
            )

    def test_collect_inputs_simple(
        self,
        mock_state: ExecutionState,
        mock_graph: MagicMock,
        mock_batch: MagicMock,
        mock_index: MagicMock,
    ):
        # quick hack to avoid case len(key) == 0, which will be removed in the future
        for key in nx.get_edge_attributes(
            mock_state.graph, DataFlowGraph.EdgeAttribute.KEY
        ).values():
            key.__len__ = MagicMock(return_value=1)

        # get the feature keys from the edges
        key_attr = DataFlowGraph.EdgeAttribute.KEY
        key_a = mock_graph.edges[("SOURCE_NODE", "PROCESSOR_NODE", "a")][
            key_attr
        ]
        key_b = mock_graph.edges[("SOURCE_NODE", "PROCESSOR_NODE", "b")][
            key_attr
        ]

        # collect the inputs to the processor node
        out_batch, out_index = mock_state.collect_inputs("PROCESSOR_NODE")
        # check the collected input batch and index
        assert out_batch == {
            "a": key_a.index_batch(mock_batch),
            "b": key_b.index_batch(mock_batch),
        }
        assert out_index == mock_index

    def test_collect_inputs_parent_not_ready(self, mock_state: ExecutionState):
        # make sure the ready event is not set for the parent node
        mock_state.ready["PROCESSOR_NODE"].clear()

        # cannot collect in case parent node is not ready
        with pytest.raises(AssertionError):
            mock_state.collect_inputs("AUGMENTER_NODE")

        # make sure the ready event is not set for the parent node
        mock_state.ready["PROCESSOR_NODE"].set()
        mock_state.outputs["PROCESSOR_NODE"] = MagicMock()

        # cannot collect in case parent node is not ready
        mock_state.collect_inputs("AUGMENTER_NODE")

    def test_collect_inputs_from_const_partition(
        self,
        mock_graph: MagicMock,
        mock_p_graph: nx.DiGraph,
        mock_batch: MagicMock,
        mock_index: MagicMock,
    ):
        attrs = {
            DataFlowGraph.EdgeAttribute.NAME: "a",
            DataFlowGraph.EdgeAttribute.KEY: MagicMock(),
        }
        # overwrite one of the edges going into the processor
        # node to connect to the constant node
        mock_graph.remove_edge("SOURCE_NODE", "PROCESSOR_NODE", key="a")
        mock_graph.add_edge("CONST_NODE", "PROCESSOR_NODE", key="a", **attrs)

        # create a mock execution state with the augmented graph
        mock_state = ExecutionState(
            mock_graph, mock_p_graph, mock_batch, mock_index, rank=0
        )

        # prepare constant node
        const_value = MagicMock()
        mock_state.outputs["CONST_NODE"] = const_value
        mock_state.ready["CONST_NODE"].set()

        # quick hack to avoid case len(key) == 0, which will be removed in the future
        for key in nx.get_edge_attributes(
            mock_state.graph, DataFlowGraph.EdgeAttribute.KEY
        ).values():
            key.__len__ = MagicMock(return_value=1)

        # get the feature keys from the edges
        key_attr = DataFlowGraph.EdgeAttribute.KEY
        key_a = mock_graph.edges[("CONST_NODE", "PROCESSOR_NODE", "a")][
            key_attr
        ]
        key_b = mock_graph.edges[("SOURCE_NODE", "PROCESSOR_NODE", "b")][
            key_attr
        ]

        # collect the inputs to the processor node
        out_batch, out_index = mock_state.collect_inputs("PROCESSOR_NODE")
        assert out_batch == {
            "a": key_a.index_batch(const_value) * len(mock_index),
            "b": key_b.index_batch(mock_batch),
        }
        assert out_index == mock_index

    def test_collect_inputs_from_parent_partition(
        self,
        mock_graph: MagicMock,
        mock_p_graph: nx.DiGraph,
        mock_batch: MagicMock,
        mock_index: MagicMock,
    ):
        attrs = {
            DataFlowGraph.EdgeAttribute.NAME: "a",
            DataFlowGraph.EdgeAttribute.KEY: MagicMock(),
        }
        # overwrite one of the edges going into the processor
        # node to connect to the constant node
        mock_graph.remove_edge(
            "AUGMENTER_NODE", "PROCESSOR_NODE_IN_AUGMENTER_PARTITION", key="a"
        )
        mock_graph.add_edge(
            "SOURCE_NODE",
            "PROCESSOR_NODE_IN_AUGMENTER_PARTITION",
            key="a",
            **attrs,
        )

        # create a mock execution state with the augmented graph
        mock_state = ExecutionState(
            mock_graph, mock_p_graph, mock_batch, mock_index, rank=0
        )

        # prepare augmenter node
        output_value = MagicMock()
        mock_state.outputs["AUGMENTER_NODE"] = output_value
        mock_state.ready["AUGMENTER_NODE"].set()
        # prepare augmenter partition
        mock_augmenter_index = MagicMock()
        mock_state.index["AUGMENTER_NODE"] = mock_augmenter_index

        # quick hack to avoid case len(key) == 0, which will be removed in the future
        for key in nx.get_edge_attributes(
            mock_state.graph, DataFlowGraph.EdgeAttribute.KEY
        ).values():
            key.__len__ = MagicMock(return_value=1)

        # get the feature keys from the edges
        key_attr = DataFlowGraph.EdgeAttribute.KEY
        key_a = mock_graph.edges[
            ("SOURCE_NODE", "PROCESSOR_NODE_IN_AUGMENTER_PARTITION", "a")
        ][key_attr]
        key_b = mock_graph.edges[
            ("AUGMENTER_NODE", "PROCESSOR_NODE_IN_AUGMENTER_PARTITION", "b")
        ][key_attr]

        mock_traced_value = MagicMock()
        with patch(
            "hyped.data.flow.core.executor.ExecutionState.trace_through_partition_path",
            MagicMock(return_value=[mock_traced_value]),
        ) as mock_trace_fn:
            # collect values
            out_batch, out_index = mock_state.collect_inputs(
                "PROCESSOR_NODE_IN_AUGMENTER_PARTITION"
            )
            assert out_index == mock_augmenter_index
            # make sure the values from the source partition are traced through the partition path
            mock_trace_fn.assert_called_once_with(
                [key_a.index_batch(mock_batch)],
                src=DataFlowGraph.PredefinedPartition.DEFAULT,
                tgt="AUGMENTER_NODE",
            )
            # check the collected batch
            assert out_batch == {
                "a": mock_traced_value,
                "b": key_b.index_batch(output_value),
            }

    def test_capture_outputs(self, mock_state: ExecutionState):
        mock_output = MagicMock()
        mock_state.capture_output("PROCESSOR_NODE", mock_output)

        assert mock_state.ready["PROCESSOR_NODE"].is_set()
        assert mock_state.outputs["PROCESSOR_NODE"] == mock_output

        # node output already set
        with pytest.raises(AssertionError):
            mock_state.capture_output("PROCESSOR_NODE", mock_output)


class TestDataFlowExecutor:
    @pytest.fixture
    def mock_collect_ref(self) -> MagicMock:
        mock_collect = MagicMock()
        mock_collect.feature_ = Features()
        return mock_collect

    @pytest.fixture
    def mock_execution_state(self) -> MagicMock:
        mock_state = MagicMock()
        mock_state.wait_for = AsyncMock()
        mock_state.collect_inputs = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )
        return mock_state

    @pytest.fixture
    def mock_node_obj(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_executor(
        self, mock_node_obj: MagicMock, mock_collect_ref: MagicMock
    ) -> DataFlowExecutor:
        expected_output = MagicMock()
        # create mock node
        node_id = "NODE_ID"
        node_obj = MagicMock()
        node_obj.get_const_batch = MagicMock(return_value=expected_output)

        mock_in_features = MagicMock()
        mock_out_features = MagicMock()
        # create mock data flow graph
        mock_G = MagicMock()
        mock_G.in_degree = MagicMock(return_value=0)
        mock_G.nodes = {
            node_id: {
                DataFlowGraph.NodeAttribute.NODE_TYPE: None,
                DataFlowGraph.NodeAttribute.NODE_OBJ: mock_node_obj,
                DataFlowGraph.NodeAttribute.IN_FEATURES: mock_in_features,
                DataFlowGraph.NodeAttribute.OUT_FEATURES: mock_out_features,
            }
        }

        aggregation_manager = MagicMock()
        aggregation_manager.aggregate = AsyncMock()

        # create the executor
        return DataFlowExecutor(
            graph=mock_G,
            collect=mock_collect_ref,
            aggregation_manager=aggregation_manager,
        )

    def test_error_on_invalid_init_args(self, mock_collect_ref):
        # initialize data flow executor with valid arguments
        DataFlowExecutor(
            graph=MagicMock(),
            collect=mock_collect_ref,
            aggregation_manager=None,
        )

        # make the mock collect feature reference invalid
        mock_collect_ref.feature_ = Value("int32")

        with pytest.raises(TypeError):
            # initialize with invalud arguments
            executor = DataFlowExecutor(
                graph=MagicMock(),
                collect=mock_collect_ref,
                aggregation_manager=None,
            )

    @pytest.mark.asyncio
    async def test_execute_node(
        self,
        mock_executor: DataFlowExecutor,
        mock_execution_state: MagicMock,
        mock_node_obj: MagicMock,
    ):
        # create a set of mock dependencies
        mock_dependencies = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        # update the mock graph to include the dependencies
        mock_executor.graph.in_degree = MagicMock(return_value=2)
        mock_executor.graph.predecessors = MagicMock(
            return_value=mock_dependencies
        )

        # execute the constant node
        await mock_executor.execute_node("NODE_ID", mock_execution_state)

        # make sure all dependencies were awaited
        mock_execution_state.wait_for.assert_has_calls(
            [call(dep) for dep in mock_dependencies], any_order=True
        )

    @pytest.mark.asyncio
    async def test_execute_const_node(
        self,
        mock_executor: DataFlowExecutor,
        mock_execution_state: MagicMock,
        mock_node_obj: MagicMock,
    ):
        # set node type
        mock_executor.graph.nodes["NODE_ID"][
            DataFlowGraph.NodeAttribute.NODE_TYPE
        ] = DataFlowGraph.NodeType.CONST

        # define mock processor object
        expected_output = MagicMock()
        mock_node_obj.get_const_batch = MagicMock(return_value=expected_output)

        # execute the constant node
        await mock_executor.execute_node("NODE_ID", mock_execution_state)

        # check if the const node output is captured correctly
        mock_node_obj.get_const_batch.assert_called_once_with(batch_size=1)
        mock_execution_state.capture_output.assert_called_once_with(
            "NODE_ID", expected_output
        )

    @pytest.mark.asyncio
    async def test_execute_processor_node(
        self,
        mock_executor: DataFlowExecutor,
        mock_execution_state: MagicMock,
        mock_node_obj: MagicMock,
    ):
        # set node type
        node_attrs = mock_executor.graph.nodes["NODE_ID"]
        node_attrs[
            DataFlowGraph.NodeAttribute.NODE_TYPE
        ] = DataFlowGraph.NodeType.DATA_PROCESSOR

        # define mock processor object
        expected_output = MagicMock()
        mock_node_obj.batch_process = AsyncMock(return_value=expected_output)

        # execute the constant node
        await mock_executor.execute_node("NODE_ID", mock_execution_state)

        mock_inputs, mock_index = mock_execution_state.collect_inputs()
        # check if the const node output is captured correctly
        mock_node_obj.batch_process.assert_called_once_with(
            mock_inputs,
            mock_index,
            mock_execution_state.rank,
            IOContext(
                node_id="NODE_ID",
                inputs=node_attrs[DataFlowGraph.NodeAttribute.IN_FEATURES],
                outputs=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURES],
            ),
        )
        mock_execution_state.capture_output.assert_called_once_with(
            "NODE_ID", expected_output
        )

    @pytest.mark.asyncio
    async def test_execute_augmenter_node(
        self,
        mock_executor: DataFlowExecutor,
        mock_execution_state: MagicMock,
        mock_node_obj: MagicMock,
    ):
        # set node type
        node_attrs = mock_executor.graph.nodes["NODE_ID"]
        node_attrs[
            DataFlowGraph.NodeAttribute.NODE_TYPE
        ] = DataFlowGraph.NodeType.DATA_AUGMENTER

        # define mock processor object
        expected_output = MagicMock()
        expected_trace_index = MagicMock()
        mock_node_obj.batch_process = AsyncMock(
            return_value=(expected_output, expected_trace_index)
        )

        # execute the constant node
        await mock_executor.execute_node("NODE_ID", mock_execution_state)

        mock_inputs, mock_index = mock_execution_state.collect_inputs()
        # check if the const node output is captured correctly
        mock_node_obj.batch_process.assert_called_once_with(
            mock_inputs,
            mock_index,
            mock_execution_state.rank,
            IOContext(
                node_id="NODE_ID",
                inputs=node_attrs[DataFlowGraph.NodeAttribute.IN_FEATURES],
                outputs=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURES],
            ),
        )
        mock_execution_state.capture_output.assert_called_once_with(
            "NODE_ID", expected_output
        )
        mock_execution_state.register_partition_trace.assert_called_once_with(
            "NODE_ID", expected_trace_index, mock_index
        )
        mock_execution_state.capture_output.assert_called_once_with(
            "NODE_ID", expected_output
        )

    @pytest.mark.asyncio
    async def test_execute_aggregater_node(
        self,
        mock_executor: DataFlowExecutor,
        mock_execution_state: MagicMock,
        mock_node_obj: MagicMock,
    ):
        # set node type
        node_attrs = mock_executor.graph.nodes["NODE_ID"]
        node_attrs[
            DataFlowGraph.NodeAttribute.NODE_TYPE
        ] = DataFlowGraph.NodeType.DATA_AGGREGATOR

        # execute the constant node
        await mock_executor.execute_node("NODE_ID", mock_execution_state)

        mock_inputs, mock_index = mock_execution_state.collect_inputs()
        # check call to aggregation manager
        mock_executor.aggregation_manager.aggregate.assert_called_once_with(
            mock_node_obj,
            mock_inputs,
            mock_index,
            mock_execution_state.rank,
            IOContext(
                node_id="NODE_ID",
                inputs=node_attrs[DataFlowGraph.NodeAttribute.IN_FEATURES],
                outputs=node_attrs[DataFlowGraph.NodeAttribute.OUT_FEATURES],
            ),
        )

    @pytest.mark.asyncio
    async def test_execute(
        self, mock_executor: DataFlowExecutor, mock_collect_ref: MagicMock
    ):
        nodes = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        # patch mock executor
        mock_executor.graph.src_node_id = nodes[0]
        mock_executor.graph.nodes = MagicMock(return_value=nodes)
        mock_executor.execute_node = AsyncMock()

        with patch(
            "hyped.data.flow.core.executor.ExecutionState"
        ) as mock_execution_state:
            output = await mock_executor.execute(
                MagicMock(), MagicMock(), MagicMock()
            )
            # make sure all nodes were awaited
            mock_executor.execute_node.assert_has_calls(
                [call(node, mock_execution_state()) for node in nodes[1:]],
                any_order=True,
            )
            # check the output
            assert output == mock_execution_state().collect_value(
                mock_collect_ref
            )
