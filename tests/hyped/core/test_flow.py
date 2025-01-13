from typing import Hashable
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import datasets
import matplotlib.pyplot as plt
import networkx as nx
import pytest

from hyped.core.features.dtypes import BoolType, Int16Type, Int32Type, MappingType
from hyped.core.features.features import Feature, MappingFeature
from hyped.core.features.reference import ConcreteReference
from hyped.core.flow import DataFlow, ExecutableDataFlow, plot_data_flow
from hyped.core.graph import DataFlowGraph
from hyped.core.optim import DataFlowGraphOptimizer
from hyped.core.typing import Bool, Mapping

from .utils import build_graph


def test_plot_data_flow():
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
    for edge in flow._graph.edges(keys=True):
        _, _, key = edge
        edge_label = "{name}".format(name=key)

        assert any(
            edge_label in text.get_text() for text in ax.texts
        ), f"Edge label {edge_label} is missing in the plot."


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
            _ = DataFlow().source

        with pytest.raises(RuntimeError):
            # hf features are not compatible with annotation
            features = datasets.Features({"other": datasets.Value("bool")})
            _ = DataFlow[SourceFeature](features).source

    @patch("hyped.core.flow.DataFlowGraph", MagicMock())
    @patch("hyped.core.flow.build_feature_from_reference", MagicMock())
    def test_const(self) -> None:
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
        with patch("hyped.core.flow.TypeAdapterWithArbitraryTypesAllowed") as mock_type_adapter:
            dtype = MagicMock()
            flow.const(42, dtype)
            # make sure the type adapter was called
            mock_type_adapter.assert_called_once_with(dtype)
            # make sure the constant was added to the graph as expected
            flow._graph.add_const_node.assert_called_once_with(
                42, mock_type_adapter.return_value.validate_python.return_value.dtype
            )

    @patch("hyped.core.flow.DataFlowGraph", MagicMock())
    @patch("hyped.core.flow.build_feature_from_reference", MagicMock())
    def test_collect(self) -> None:
        # create a data flow instance and a mock feature
        flow = DataFlow()
        feature = MagicMock(spec=Feature, ref=MagicMock(spec=ConcreteReference))

        # test trivial case
        assert feature == flow.collect(feature)

        flow.collect({"x": feature})
        flow._graph.add_collect_node_with_constants.assert_called_once_with({"x": feature.ref})

    @patch("hyped.core.flow.DataFlowGraph")
    @patch("hyped.core.flow.ExecutableDataFlow")
    @patch("hyped.core.flow.build_feature_from_reference", MagicMock())
    def test_build(self, mock_executable_flow: MagicMock, mock_graph: MagicMock) -> None:
        # set source node id of mock graph to none
        # to mimic the flow not being initialized
        flow = DataFlow(datasets.Features({"field": datasets.Value("bool")}))
        flow._graph.src_node_id = None

        # create valid and invalid features
        valid = MagicMock(
            spec=Feature,
            ref=ConcreteReference(_node_id=MagicMock(), _graph=mock_graph.return_value),
        )
        invalid = MagicMock(
            spec=Feature, ref=ConcreteReference(_node_id=MagicMock(), _graph=MagicMock())
        )

        # invalid collect feature, not belonging to the graph
        with pytest.raises(RuntimeError):
            flow.build(collect=invalid)

        # invalid aggregate feature, not belonging to the graph
        with pytest.raises(RuntimeError):
            flow.build(collect=valid, aggregate=invalid)

        manager = MagicMock()

        with (
            patch("hyped.core.flow.DataFlow._initialize"),
            patch(
                "hyped.core.flow.DataFlow._source_annotation", PropertyMock()
            ) as mock_source_annotation,
            patch("hyped.core.features.features.MappingFeature.__post_init__"),
            patch("hyped.core.flow.nx.restricted_view") as mock_restricted_view,
        ):
            flow.build(collect=valid, aggregate=valid, aggregation_manager=manager)

            mock_restricted_view.assert_called_once_with(mock_graph.return_value, [], [])
            mock_executable_flow.assert_called_once_with(
                mock_source_annotation.return_value,
                mock_restricted_view.return_value,
                ConcreteReference(valid.ref._node_id, mock_restricted_view.return_value),
                ConcreteReference(valid.ref._node_id, mock_restricted_view.return_value),
                manager,
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

        with patch("hyped.core.flow.build_feature_from_reference", MagicMock(spec=MappingFeature)):
            with pytest.raises(RuntimeError):
                # invalid node id in collect reference
                collect = MagicMock(spec=ConcreteReference, _graph=mock_graph, _node_id=MagicMock())
                ExecutableDataFlow(None, mock_graph, collect, None, None)

            with pytest.raises(RuntimeError):
                # collect node cannot be part of aggregated partition
                collect = MagicMock(
                    spec=ConcreteReference, _graph=mock_graph, _node_id=mock_aggregate_node._node_id
                )
                ExecutableDataFlow(None, mock_graph, collect, None, None)

            # valid collect reference
            collect = MagicMock(
                spec=ConcreteReference, _graph=mock_graph, _node_id=mock_feature_node._node_id
            )

            with pytest.raises(RuntimeError):
                # invalid node id in collect reference
                aggregate = MagicMock(
                    spec=ConcreteReference, _graph=mock_graph, _node_id=MagicMock()
                )
                ExecutableDataFlow(None, mock_graph, collect, aggregate, None)

            with pytest.raises(RuntimeError):
                # aggregate node must be part of aggregated partition
                aggregate = MagicMock(
                    spec=ConcreteReference, _graph=mock_graph, _node_id=mock_feature_node._node_id
                )
                ExecutableDataFlow(None, mock_graph, collect, aggregate, None)

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
                build_graph([(0, 1), (1, 2)]),  # 1 is getitem operator
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
        collect = ConcreteReference(_node_id=collect, _graph=graph) if collect is not None else None
        aggregate = (
            ConcreteReference(_node_id=aggregate, _graph=graph) if aggregate is not None else None
        )

        mock_optimizer = MagicMock(spec=DataFlowGraphOptimizer)
        mock_optimizer.optimize.return_value = graph

        with (
            patch("hyped.core.flow.DataFlowGraphOptimizer", MagicMock(return_value=mock_optimizer)),
            patch("hyped.core.flow.DataFlowExecutor") as mock_executor,
            patch("hyped.core.flow.LazyDataFlowExecutor") as mock_lazy_executor,
            patch("hyped.core.flow.DataAggregationManager"),
        ):
            # create the executable flow
            flow = ExecutableDataFlow(None, graph, collect, aggregate, None)

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

    @patch("hyped.core.flow.datasets.fingerprint.generate_fingerprint", MagicMock())
    @patch("hyped.core.flow.datasets.fingerprint.update_fingerprint", MagicMock())
    def test_apply(self) -> None:
        # create a simple data flow graph containing only a source node
        graph = DataFlowGraph()
        graph.add_source_node(MappingType.construct({"x": BoolType}))
        # create the collect reference
        collect = ConcreteReference(_node_id=graph.src_node_id, _graph=graph)
        # create the executable data flow instance
        flow = ExecutableDataFlow(None, graph, collect, None, None)

        with pytest.raises(ValueError):
            # not a dataset
            flow.apply(ds=MagicMock())

        # create a mock dataset
        hf_features = datasets.Features({"x": datasets.Value("bool")})
        ds = MagicMock(spec=datasets.Dataset, features=hf_features)
        # apply the flow to the mock dataset
        flow.apply(ds)
        # make sure the map function was called correctly
        ds.with_format.assert_called_once_with(type="arrow", columns=["x"])
        ds.with_format.return_value.map.assert_called_once_with(
            flow._instance_executor.run,
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

        ds.reset_mock()

        # create a mock dataset dict
        ds_dict = datasets.DatasetDict({"data": ds})
        # apply the flow to the mock dataset
        flow.apply(ds_dict)
        # make sure the map function was called correctly
        ds.with_format.assert_called_once_with(type="arrow", columns=["x"])
        ds.with_format.return_value.map.assert_called_once_with(
            flow._instance_executor.run,
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
            flow._instance_executor.run,
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
            flow._instance_executor.run,
            with_indices=True,
            batched=True,
            batch_size=ANY,
            drop_last_batch=ANY,
            remove_columns=ANY,
        )

    @patch("hyped.core.flow.datasets.fingerprint.generate_fingerprint", MagicMock())
    @patch("hyped.core.flow.datasets.fingerprint.update_fingerprint", MagicMock())
    def test_apply_fallback_on_feature_mismatch(self) -> None:
        # create two different but castable dtypes
        dtype_A = MappingType.construct({"field": Int16Type})
        dtype_B = MappingType.construct({"field": Int32Type})
        # create a simple data flow graph containing only a source node
        # using the first dtype and a single aggregator node
        from hyped.core.ops.mapping import MappingGetItem
        from hyped.core.ops.numeric import Sum

        graph = DataFlowGraph()
        src_ref = graph.add_source_node(dtype_A)
        val_ref = graph.add_compute_node(MappingGetItem(key="field"), {"mapping": src_ref})
        sum_ref = graph.add_compute_node(Sum(), {"val": val_ref})
        agg_ref = graph.add_collect_node({"sum": sum_ref})
        # create the executable data flow instance
        mock_manager = MagicMock(
            values_proxy={sum_ref._node_id: MagicMock(type=Int16Type.arrow_type)}
        )
        flow = ExecutableDataFlow(None, graph, src_ref, agg_ref, mock_manager)

        # create a mock dataset with features matching the second data type
        ds = MagicMock(spec=datasets.Dataset, features=dtype_B.hf_feature)

        exec_flow_buffer: list[ExecutableDataFlow] = []

        def capture_exec_flow_hook(*args, **kwargs):
            exec_flow = ExecutableDataFlow(*args, **kwargs)
            exec_flow_buffer.append(exec_flow)
            return exec_flow

        mock_exec_flow_class = MagicMock(side_effect=capture_exec_flow_hook)

        with patch("hyped.core.flow.ExecutableDataFlow", mock_exec_flow_class):
            flow.apply(ds)

        # make sure a new data flow was build by the apply
        mock_exec_flow_class.assert_called_once()
        assert len(exec_flow_buffer) == 1
        # check the structure of the exec
        (exec_flow,) = exec_flow_buffer
        # make sure the new flow uses the same aggregation manager
        assert exec_flow._aggregation_manager == mock_manager
        # make sure the source features match the features of the dataset
        assert exec_flow._graph.src_dtype == dtype_B
        # make sure there is exactly one node connecting to the source node
        assert exec_flow._graph.out_degree(exec_flow._graph.src_node_id) == 1
        edges = exec_flow._graph.out_edges(exec_flow._graph.src_node_id)
        _, cast_node_id = next(iter(edges))
        # make sure that node is the cast node
        node_type = exec_flow._graph.nodes[cast_node_id][DataFlowGraph.NodeAttribute.NODE_TYPE]
        out_dtype = exec_flow._graph.nodes[cast_node_id][
            DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE
        ]
        assert node_type == DataFlowGraph.NodeType.CAST
        assert out_dtype == dtype_A

    @patch("hyped.core.flow.datasets.fingerprint.generate_fingerprint", MagicMock())
    @patch("hyped.core.flow.datasets.fingerprint.update_fingerprint", MagicMock())
    def test_apply_exceptions(self) -> None:
        # create two different and un-castable dtypes
        dtype_A = MappingType.construct({"field": Int16Type})
        dtype_B = MappingType.construct({"other": Int32Type})
        # create a simple data flow graph containing only a source node
        # using the first dtype
        graph = DataFlowGraph()
        graph.add_source_node(dtype_A)
        # create the collect reference
        collect = ConcreteReference(_node_id=graph.src_node_id, _graph=graph)
        # create the executable data flow instance
        flow = ExecutableDataFlow(None, graph, collect, None, None)

        with pytest.raises(ValueError, match="Expected one of `datasets.Dataset`,"):
            flow.apply(MagicMock())

        # create a mock dataset with undefined dataset features
        ds = MagicMock(spec=datasets.Dataset, features=None)
        with pytest.raises(RuntimeError, match="Dataset features must not be None."):
            flow.apply(ds)

        # create a mock dataset with features matching the second data type
        ds = MagicMock(spec=datasets.Dataset, features=dtype_B.hf_feature)
        with pytest.raises(RuntimeError, match="Failed to cast dataset features"):
            flow.apply(ds)

        with (
            pytest.raises(RuntimeError, match="Dataset features do not align with the expected"),
            patch("hyped.core.flow.validate_hf_feature") as mock_validate_hf_feature,
        ):
            flow = ExecutableDataFlow(MagicMock(), graph, collect, None, None)
            mock_validate_hf_feature.side_effect = Exception
            flow.apply(ds)

    def test_serialization(self) -> None:
        class Inputs(Mapping):
            x: Bool

        # create a simple data flow graph containing only a source node
        graph = DataFlowGraph()
        graph.add_source_node(MappingType.construct({"x": BoolType}))
        # create the collect reference
        collect = ConcreteReference(_node_id=graph.src_node_id, _graph=graph)
        # create the executable data flow instance
        flow = ExecutableDataFlow(Inputs, graph, collect, None, None)

        serialized = flow.serialize()
        flow = DataFlow.deserialize(serialized)

        assert collect._node_id == flow._instance_executor.collect._node_id

        with pytest.raises(ValueError):
            # deserialize from invalid string
            DataFlow.deserialize("{}")
