import asyncio
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
from datasets import Features, Value

from hyped.data.flow.core.executor import (
    Batch,
    DataFlowExecutor,
    ExecutionState,
)
from hyped.data.flow.core.graph import DataFlowGraph

from .mock import MockAggregator, MockAugmenter, MockProcessor


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
                DataFlowGraph.NodeAttribute.NODE_OBJ: MockProcessor(),
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
                DataFlowGraph.NodeAttribute.NODE_OBJ: MockAugmenter(),
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
                DataFlowGraph.NodeAttribute.NODE_OBJ: MockProcessor(),
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
                DataFlowGraph.NodeAttribute.NODE_OBJ: MockAugmenter(),
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
                DataFlowGraph.NodeAttribute.NODE_OBJ: MockProcessor(),
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

        mock_G = MagicMock(wraps=G)
        mock_G.nodes = G.nodes
        mock_G.edges = G.edges
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
