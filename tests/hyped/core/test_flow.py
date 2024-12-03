from typing import Hashable
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import datasets
import matplotlib.pyplot as plt
import networkx as nx
import pytest

from hyped.core.executor import DataFlowExecutor
from hyped.core.features.features import Feature, MappingFeature
from hyped.core.features.reference import Reference
from hyped.core.features.types import BoolType, MappingType
from hyped.core.flow import DataFlow, ExecutableDataFlow, plot_data_flow
from hyped.core.graph import DataFlowGraph
from hyped.core.optim import DataFlowGraphOptimizer
from hyped.core.typing import Bool, Mapping

from .utils import build_graph


@pytest.mark.parametrize(
    "with_edge_labels, edge_label_format",
    [
        (False, "{name}={key}"),
        (True, "{name}={key}"),
        (True, "{name}"),
        (True, "{key}"),
    ],
)
def test_plot_data_flow(with_edge_labels, edge_label_format):
    # build a graph
    flow = DataFlow({"field": datasets.Value("bool")})
    flow._graph = build_graph(
        [(0, 1), (1, 2)],
        {
            0: DataFlowGraph.NodeType.SOURCE,
            1: DataFlowGraph.NodeType.DATA_PROCESSOR,
            2: DataFlowGraph.NodeType.DATA_PROCESSOR,
        },
        {
            0: MagicMock(),
            1: MagicMock(__class__=MagicMock(__name__="A")),
            2: MagicMock(__class__=MagicMock(__name__="A")),
        },
    )

    # Ensure the plot function runs without errors and returns an Axes object
    with patch("matplotlib.pyplot.show"):  # Mock plt.show to avoid displaying the plot during tests
        ax = plot_data_flow(
            flow,
            src_node_label="[ROOT]",
            with_edge_labels=with_edge_labels,
            node_font_size=1e-5,
        )
        assert isinstance(ax, plt.Axes)

    # Check if node labels are correct
    for node, data in flow._graph.nodes(data=True):
        node_label = (
            "[ROOT]"
            if node == flow._graph.src_node_id
            else type(data[DataFlowGraph.NodeAttribute.NODE_OBJ]).__name__
        )
        assert any(
            node_label in text.get_text() for text in ax.texts
        ), f"Node label {node_label} is missing in the plot."

    # Check if edge labels are correct
    for edge in flow._graph.edges(data=True):
        _, _, data = edge
        edge_label = edge_label_format.format(
            name=data[DataFlowGraph.EdgeAttribute.NAME],
            key=data[DataFlowGraph.EdgeAttribute.KEY],
        )

        if with_edge_labels:
            assert any(
                edge_label in text.get_text() for text in ax.texts
            ), f"Edge label {edge_label} is missing in the plot."
        else:
            assert all(
                edge_label not in text.get_text() for text in ax.texts
            ), f"Edge label {edge_label} in the plot but shouldn't be included."


class TestDataFlow:
    def test_initialize(self) -> None:
        class SourceFeature(Mapping):
            field: Bool

        hf_features = datasets.Features({"field": datasets.Value("bool")})
        hyped_type = MappingType.construct({"field": BoolType})

        # initialize from huggingface features
        flow = DataFlow(hf_features)
        assert flow.source.dtype == hyped_type

        # initialize from type annotation
        flow = DataFlow[SourceFeature]()
        assert flow.source.dtype == hyped_type

        # initialize from both
        flow = DataFlow[SourceFeature](hf_features)
        assert flow.source.dtype == hyped_type

        with pytest.raises(RuntimeError):
            # no input features specified
            DataFlow().source

        with pytest.raises(RuntimeError):
            # hf features are not compatible with annotation
            DataFlow[SourceFeature](datasets.Features({"other": datasets.Value("bool")})).source

    @patch("hyped.core.flow.DataFlowGraph")
    def test_const(self, mock_graph: MagicMock) -> None:
        # create a data flow graph instance
        flow = DataFlow()

        # add constant without specifying the data type
        with patch("hyped.core.flow.build_dtype_from_python_object") as mock_build_dtype:
            flow.const(42)
            # make sure the constant was added to the graph as expected
            flow._graph.add_const_node.assert_called_once_with(42, mock_build_dtype.return_value)

        # reset the mock graph
        flow._graph.reset_mock()

        # add constant with specified data type
        with patch("hyped.core.flow.pydantic.TypeAdapter") as mock_type_adapter:
            dtype = MagicMock()
            flow.const(42, dtype)
            # make sure the type adapter was called
            mock_type_adapter.assert_called_once_with(dtype)
            # make sure the constant was added to the graph as expected
            flow._graph.add_const_node.assert_called_once_with(
                42, mock_type_adapter.return_value.validate_python.return_value.dtype
            )

    @patch("hyped.core.flow.DataFlowGraph")
    def test_collect(self, mock_graph: MagicMock) -> None:
        # create a data flow instance and a mock feature
        flow = DataFlow()
        feature = MagicMock(spec=Feature, ref=MagicMock(spec=Reference))

        # test trivial case
        assert feature == flow.collect(feature)

        flow.collect({"x": feature})
        flow._graph.add_collect_node_with_constants.assert_called_once_with({"x": feature.ref})

    @patch("hyped.core.flow.DataFlowGraph")
    @patch("hyped.core.flow.ExecutableDataFlow")
    def test_build(self, mock_executable_flow: MagicMock, mock_graph: MagicMock) -> None:
        flow = DataFlow(datasets.Features({"field": datasets.Value("bool")}))
        # set source node id of mock graph to none
        # to mimic the flow not being initialized
        flow._graph.src_node_id = None

        # create valid and invalid features
        valid = MagicMock(
            spec=Feature, ref=MagicMock(spec=Reference, _graph=mock_graph.return_value)
        )
        invalid = MagicMock(spec=Feature, ref=MagicMock(spec=Reference, _graph=MagicMock()))

        # invalid collect feature, not belonging to the graph
        with pytest.raises(RuntimeError):
            flow.build(collect=invalid)

        # invalid aggregate feature, not belonging to the graph
        with pytest.raises(RuntimeError):
            flow.build(collect=valid, aggregate=invalid)

        with patch("hyped.core.flow.nx.restricted_view") as mock_restricted_view:
            flow.build(collect=valid, aggregate=valid)

            mock_restricted_view.assert_called_once_with(mock_graph.return_value, [], [])

            mock_executable_flow.assert_called_once_with(
                mock_restricted_view.return_value,
                Reference(valid.ref._key, valid.ref._node_id, mock_restricted_view.return_value),
                Reference(valid.ref._key, valid.ref._node_id, mock_restricted_view.return_value),
            )

    @patch("hyped.core.flow.DataFlow.build")
    def test_apply(self, mock_build: MagicMock) -> None:
        ds = MagicMock()
        collect = MagicMock()
        aggregate = MagicMock()
        # apply a data flow to a dataset
        DataFlow().apply(ds, collect, aggregate)
        # make sure the flow was build and the executable flow was applied
        mock_build.assert_called_once_with(collect, aggregate)
        mock_build.return_value.apply.assert_called_once_with(ds)


class TestExecutableDataFlow:
    def test_initialize_validation(self) -> None:
        mock_feature_node = MagicMock()
        mock_aggregate_node = MagicMock()

        mock_graph = MagicMock(
            spec=DataFlowGraph, nodes=[mock_feature_node._node_id, mock_aggregate_node._node_id]
        )
        mock_graph.get_node_output_partition.side_effect = {
            mock_feature_node._node_id: DataFlowGraph.Partition.DEFAULT,
            mock_aggregate_node._node_id: DataFlowGraph.Partition.AGGREGATED,
        }.get
        mock_graph.get_feature_from_reference.return_value = MagicMock(spec=MappingFeature)

        with pytest.raises(RuntimeError):
            # invalid node id in collect reference
            collect = MagicMock(spec=Reference, _graph=mock_graph, _node_id=MagicMock())
            ExecutableDataFlow(mock_graph, collect, None)

        with pytest.raises(RuntimeError):
            # collect node cannot be part of aggregated partition
            collect = MagicMock(
                spec=Reference, _graph=mock_graph, _node_id=mock_aggregate_node._node_id
            )
            ExecutableDataFlow(mock_graph, collect, None)

        # valid collect reference
        collect = MagicMock(spec=Reference, _graph=mock_graph, _node_id=mock_feature_node._node_id)

        with pytest.raises(RuntimeError):
            # invalid node id in collect reference
            aggregate = MagicMock(spec=Reference, _graph=mock_graph, _node_id=MagicMock())
            ExecutableDataFlow(mock_graph, collect, aggregate)

        with pytest.raises(RuntimeError):
            # aggregate node must be part of aggregated partition
            aggregate = MagicMock(
                spec=Reference, _graph=mock_graph, _node_id=mock_feature_node._node_id
            )
            ExecutableDataFlow(mock_graph, collect, aggregate)

    @pytest.mark.parametrize(
        "graph, collect, aggregate, expected_instance_graph, expected_aggregates_graph",
        [
            # Simple linear graph with no aggregators
            (
                build_graph(
                    [(0, 1), (1, 2)],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    output_type=MappingType.construct({"x": BoolType}),
                ),
                2,
                None,
                build_graph([(0, 1), (1, 2)]),
                None,
            ),
            # Simple linear graph with aggregators
            (
                build_graph(
                    [(0, 1), (1, 2)],
                    {
                        0: DataFlowGraph.NodeType.SOURCE,
                        1: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                        2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    },
                    output_type=MappingType.construct({"x": BoolType}),
                ),
                0,
                2,
                build_graph([(0, 1)]),  # aggregator is contained in instance graph
                build_graph([(0, 1)]),
            ),
        ],
    )
    def test_initialize(
        self,
        graph: DataFlowGraph,
        collect: Hashable,
        aggregate: Hashable,
        expected_instance_graph: DataFlowGraph,
        expected_aggregates_graph: DataFlowGraph,
    ) -> None:
        collect = Reference(_node_id=collect, _graph=graph) if collect is not None else None
        aggregate = Reference(_node_id=aggregate, _graph=graph) if aggregate is not None else None

        mock_optimizer = MagicMock(spec=DataFlowGraphOptimizer)
        mock_optimizer.optimize.return_value = graph

        with (
            patch("hyped.core.flow.DataFlowGraphOptimizer", MagicMock(return_value=mock_optimizer)),
            patch("hyped.core.flow.DataFlowExecutor") as mock_executor,
            patch("hyped.core.flow.LazyDataFlowExecutor") as mock_lazy_executor,
            patch("hyped.core.flow.DataAggregationManager") as mock_aggregation_manager,
        ):
            # create the executable flow
            flow = ExecutableDataFlow(graph, collect, aggregate)

            # make sure the instance graph has the expected structure
            assert nx.is_isomorphic(flow._instance_graph, expected_instance_graph)
            assert flow._instance_executor == mock_executor.return_value

            if expected_aggregates_graph is not None:
                # make sure the aggregates graph has the expected structure
                assert nx.is_isomorphic(flow._aggregates_graph, expected_aggregates_graph)
                assert flow._aggregates_executor == mock_lazy_executor.return_value
            else:
                # make sure there is no aggregates graph
                assert flow._aggregates_graph is None
                assert flow._aggregates_executor is None

    def test_pyarrow_process(self) -> None:
        # create a simple data flow graph containing only a source node
        graph = DataFlowGraph()
        graph.add_source_node(MappingType.construct({"x": BoolType}))
        # create the collect reference
        collect = Reference(_node_id=graph.src_node_id, _graph=graph)

        mock_executor = MagicMock(spec=DataFlowExecutor, execute=AsyncMock())
        # mock the executor
        with (
            patch("hyped.core.flow.pa.table") as mock_table,
            patch("hyped.core.flow.get_worker_info") as mock_get_worker_info,
            patch("hyped.core.flow.DataFlowExecutor", MagicMock(return_value=mock_executor)),
        ):
            # create the executable data flow instance
            flow = ExecutableDataFlow(graph, collect, None)

            # create mock inputs to be processed
            mock_batch = MagicMock()
            mock_index = MagicMock()
            mock_rank = MagicMock()

            # test pyarrow process call
            out = flow.pyarrow_process(mock_batch, mock_index, mock_rank)
            mock_executor.execute.assert_called_once_with(
                mock_batch.to_struct_array.return_value, mock_index, mock_rank
            )
            assert out == mock_table.return_value

            # reset mocks
            mock_get_worker_info.reset_mock()
            mock_executor.reset_mock()

            # test infer default rank from worker info
            flow.pyarrow_process(mock_batch, mock_index, None)
            mock_get_worker_info.assert_called_once()
            # check execute function called correctly
            mock_executor.execute.assert_called_once_with(
                mock_batch.to_struct_array.return_value,
                mock_index,
                mock_get_worker_info.return_value.rank,
            )
            assert out == mock_table.return_value

    @patch("hyped.core.flow.datasets.fingerprint.generate_fingerprint", MagicMock())
    @patch("hyped.core.flow.datasets.fingerprint.update_fingerprint", MagicMock())
    def test_apply(self) -> None:
        # create a simple data flow graph containing only a source node
        graph = DataFlowGraph()
        graph.add_source_node(MappingType.construct({"x": BoolType}))
        # create the collect reference
        collect = Reference(_node_id=graph.src_node_id, _graph=graph)
        # create the executable data flow instance
        flow = ExecutableDataFlow(graph, collect, None)

        with pytest.raises(ValueError):
            # not a dataset
            flow.apply(ds=MagicMock())

        # create a mock dataset
        hf_features = datasets.Features({"x": datasets.Value("bool")})
        ds = MagicMock(spec=datasets.Dataset, features=hf_features)
        # apply the flow to the mock dataset
        flow.apply(ds)
        # make sure the map function was called correctly
        ds.with_format.assert_called_once_with(type="arrow", columns=[])
        ds.with_format.return_value.map.assert_called_once_with(
            flow.pyarrow_process,
            with_indices=True,
            with_rank=True,
            batched=True,
            batch_size=ANY,
            drop_last_batch=ANY,
            keep_in_memory=ANY,
            load_from_cache_file=ANY,
            writer_batch_size=ANY,
            num_proc=ANY,
            desc=ANY,
            new_fingerprint=ANY,
        )

        with patch("hyped.core.flow.is_dtype_subset", MagicMock(return_value=False)), pytest.raises(
            RuntimeError
        ):
            # required source dtype is not a subset of the dataset
            flow.apply(ds)

        ds.reset_mock()

        # create a mock dataset dict
        ds_dict = datasets.DatasetDict({"data": ds})
        # apply the flow to the mock dataset
        flow.apply(ds_dict)
        # make sure the map function was called correctly
        ds.with_format.assert_called_once_with(type="arrow", columns=[])
        ds.with_format.return_value.map.assert_called_once_with(
            flow.pyarrow_process,
            with_indices=True,
            with_rank=True,
            batched=True,
            batch_size=ANY,
            drop_last_batch=ANY,
            keep_in_memory=ANY,
            load_from_cache_file=ANY,
            writer_batch_size=ANY,
            num_proc=ANY,
            desc=ANY,
            new_fingerprint=ANY,
        )

        # create a mock iterable dataset
        it_ds = MagicMock(spec=datasets.IterableDataset, features=hf_features)
        # apply the flow to the mock dataset
        flow.apply(it_ds)
        # make sure the map function was called correctly
        it_ds.with_format.assert_called_once_with(type="arrow")
        it_ds.with_format.return_value.map.assert_called_once_with(
            flow.pyarrow_process,
            with_indices=True,
            batched=True,
            batch_size=ANY,
            drop_last_batch=ANY,
            remove_columns=ANY,
            features=ANY,
        )

        # create a mock iterable dataset
        it_ds_dict = MagicMock(
            spec=datasets.IterableDatasetDict, values=MagicMock(return_value=[it_ds])
        )
        # apply the flow to the mock dataset
        flow.apply(it_ds_dict)
        # make sure the map function was called correctly
        it_ds_dict.with_format.assert_called_once_with(type="arrow")
        it_ds_dict.with_format.return_value.map.assert_called_once_with(
            flow.pyarrow_process,
            with_indices=True,
            batched=True,
            batch_size=ANY,
            drop_last_batch=ANY,
            remove_columns=ANY,
        )
