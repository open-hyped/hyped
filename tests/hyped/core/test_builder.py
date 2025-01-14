from typing import Hashable
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from hyped.core.builder import DataFlowGraphBuilder
from hyped.core.features.dtypes import BoolType as MockType
from hyped.core.features.dtypes import MappingType, SequenceType
from hyped.core.features.features import build_feature_from_reference
from hyped.core.features.reference import ConcreteReference
from hyped.core.graph import DataFlowGraph
from hyped.core.nodes.augmentor import BaseDataAugmentor
from hyped.core.nodes.const import ConstNode
from hyped.core.typing import PartitionId

from .utils import build_graph


class TestDataFlowGraphBuilder:
    @pytest.mark.parametrize(
        "edges,, node_types, node_id, expected_partition, raises_error",
        [
            # Source node is always in default partition
            (
                [(0, 1)],
                {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                0,
                DataFlowGraph.Partition.DEFAULT,
                False,
            ),
            # Data Processors don't change the partition
            (
                [(0, 1)],
                {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                1,
                DataFlowGraph.Partition.DEFAULT,
                False,
            ),
            # Constants are always in the constant partition
            (
                [(0, 1), (2, 1)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.CONST,
                },
                2,
                DataFlowGraph.Partition.CONST,
                False,
            ),
            # Default partition wins when mixing with the constant partition
            (
                [(0, 1), (2, 1)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.CONST,
                },
                1,
                DataFlowGraph.Partition.DEFAULT,
                False,
            ),
            # Any node with only constant inputs is in the constant partition
            (
                [(0, 1), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.CONST,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                3,
                DataFlowGraph.Partition.CONST,
                False,
            ),
            # Aggregators are not part of the aggregated partition
            (
                [(0, 1), (1, 2)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                1,
                DataFlowGraph.Partition.DEFAULT,
                False,
            ),
            # Aggregators always map into the aggregated partition
            (
                [(0, 1), (1, 2)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                2,
                DataFlowGraph.Partition.AGGREGATED,
                False,
            ),
            # Augmentors are not part of their own partition
            (
                [(0, 1), (1, 2)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                1,
                DataFlowGraph.Partition.DEFAULT,
                False,
            ),
            # Augmentors introduce a new partition
            (
                [(0, 1), (1, 2)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                2,
                1,  # partition uses the same id as the augmentor node
                False,
            ),
            # Chaining augmentors the latest augmentor partition wins
            (
                [(0, 1), (1, 2), (2, 3), (1, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    3: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                3,
                2,
                False,
            ),
            # Cannot mix independent augmentator partitions
            (
                [(0, 1), (0, 2), (2, 3), (1, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    3: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                3,
                None,
                True,
            ),
            # Cannot mix aggregated with non-aggregated features
            (
                [(0, 1), (0, 2), (2, 3), (1, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                3,
                None,
                True,
            ),
        ],
    )
    def test_infer_node_partition(
        self,
        edges: list[tuple[Hashable, Hashable]],
        node_types: dict[Hashable, DataFlowGraph.NodeType],
        node_id: Hashable,
        expected_partition: None | PartitionId,
        raises_error: bool,
    ) -> None:
        mock_augmentor_nodes = {
            i: MagicMock(
                __spec__=BaseDataAugmentor, infer_output_partition=MagicMock(return_value=i)
            )
            for i, node_type in node_types.items()
            if node_type == DataFlowGraph.NodeType.DATA_AUGMENTOR
        }

        # build the data flow graph and create the builder instance
        graph = build_graph(edges, node_types, mock_augmentor_nodes, stop_at_node=node_id)
        builder = DataFlowGraphBuilder(graph)

        # get the node type and build the reference instances
        node_type = node_types[node_id]
        in_nodes = [u for u, v in edges if v == node_id]

        if raises_error:
            with pytest.raises(RuntimeError):
                builder._infer_node_partition(node_type, in_nodes)
        else:
            partition = builder._infer_node_partition(node_type, in_nodes)
            assert partition == expected_partition

    @pytest.mark.parametrize(
        "edges, nodes, partition_edges",
        [
            # Linear graph of data processors
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                [],
            ),
            # Linear graph including aggregator nodes
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                [(DataFlowGraph.Partition.DEFAULT.value, DataFlowGraph.Partition.AGGREGATED.value)],
            ),
            # Linear graph including augmentor and aggregator nodes
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    3: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                },
                [
                    (DataFlowGraph.Partition.DEFAULT.value, 2),
                    (2, DataFlowGraph.Partition.AGGREGATED),
                ],
            ),
            # Linear graph with multiple augmentors
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    3: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                },
                [(DataFlowGraph.Partition.DEFAULT.value, 1), (1, 2), (2, 3)],
            ),
            # Tree with two branches
            (
                [(0, 1), (0, 2), (1, 3), (2, 4)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    4: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                },
                [
                    (DataFlowGraph.Partition.DEFAULT, 1),
                    (DataFlowGraph.Partition.DEFAULT, 2),
                    (2, DataFlowGraph.Partition.AGGREGATED),
                ],
            ),
            # DAG with only default partition
            (
                [(0, 1), (0, 2), (2, 3), (1, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                [],
            ),
            # DAG with constant node
            (
                [(0, 1), (2, 1)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.CONST,
                },
                [],  # constant partition is not connected to other partitions
            ),
        ],
    )
    def test_build_partition_graph(
        self,
        edges: list[tuple[Hashable, Hashable]],
        nodes: dict[Hashable, DataFlowGraph.NodeType],
        partition_edges: list[tuple[PartitionId, PartitionId]],
    ) -> None:
        mock_augmentor_nodes = {
            i: MagicMock(
                __spec__=BaseDataAugmentor, infer_output_partition=MagicMock(return_value=i)
            )
            for i, node_type in nodes.items()
            if node_type == DataFlowGraph.NodeType.DATA_AUGMENTOR
        }

        graph = build_graph(edges, nodes, mock_augmentor_nodes)
        builder = DataFlowGraphBuilder(graph)
        partition = builder._build_partition_graph()

        target_partition_graph = nx.DiGraph()
        target_partition_graph.add_nodes_from(
            [
                DataFlowGraph.Partition.CONST.value,
                DataFlowGraph.Partition.DEFAULT.value,
            ]
        )

        target_partition_graph.add_edges_from(partition_edges)

        assert nx.utils.misc.graphs_equal(partition, target_partition_graph)

    @patch("hyped.core.builder.build_dtype_from_python_object")
    def test_add_consts_from_nested(self, mock_dtype_from_python: MagicMock) -> None:
        # create a builder instance
        builder = DataFlowGraphBuilder()
        builder.const = MagicMock()
        # add a source node
        ref = builder.source(MagicMock(spec=MappingType))

        builder.const.reset_mock()
        # nothing to add
        out = builder._add_consts_from_nested({"x": ref})
        builder.const.assert_not_called()
        assert out == {"x": ref}

        builder.const.reset_mock()
        # add constant
        out = builder._add_consts_from_nested({"x": ref, "y": 42})
        builder.const.assert_called_once_with(42, dtype=mock_dtype_from_python.return_value)
        assert out == {"x": ref, "y": builder.const.return_value}

        builder.const.reset_mock()
        # add sequence of constants
        out = builder._add_consts_from_nested([42, 42, 42])
        builder.const.assert_called_with(42, dtype=mock_dtype_from_python.return_value)
        assert out == [builder.const.return_value] * 3

    def test_const(self) -> None:
        builder = DataFlowGraphBuilder()

        # add the node to the graph
        ref = builder.const(False, MockType)

        # make sure node was added as expected
        attrs = builder.graph.nodes[ref._node_id]
        assert isinstance(attrs[DataFlowGraph.NodeAttribute.NODE_OBJ], ConstNode)
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.CONST
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MockType
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.construct({})

    def test_cast(self) -> None:
        builder = DataFlowGraphBuilder()
        src_ref = builder.source(MockType)
        # add cast node to graph
        ref = builder.cast(src_ref, MockType)
        # make sure node was added as expected
        attrs = builder.graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ] is None
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.CAST
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MockType
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.construct(
            {"value": MockType}
        )
        # check error handling when type casting is invalid
        with patch("hyped.core.builder.cast_dtype", MagicMock(side_effect=RuntimeError)):
            with pytest.raises(RuntimeError):
                builder.cast(src_ref, MockType)

    def test_collect(self) -> None:
        # create simple linear graph
        graph = build_graph([(0, 1), (1, 2), (2, 3)])
        builder = DataFlowGraphBuilder(graph)

        ref = builder.collect(
            {
                "a": ConcreteReference(0, graph, builder),
                "b": ConcreteReference(1, graph, builder),
            }
        )
        # check node attributes
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ].config.lookup == {"a": "a", "b": "b"}
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.COLLECT
        assert (
            attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE]
            == attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
            == MappingType.construct(
                {
                    "a": graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                    "b": graph.nodes[1][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                }
            )
        )
        # check edges
        assert (0, ref._node_id, "a") in graph.edges
        assert (1, ref._node_id, "b") in graph.edges

        ref = builder.collect(
            {
                "a": {
                    "b": ConcreteReference(0, graph, builder),
                    "c": ConcreteReference(1, graph, builder),
                },
            }
        )
        # check node attributes
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ].config.lookup == {
            "a": {"b": "a.b", "c": "a.c"}
        }
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.COLLECT
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.construct(
            {
                "a.b": graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                "a.c": graph.nodes[1][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
            }
        )
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MappingType.construct(
            {
                "a": MappingType.construct(
                    {
                        "b": graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                        "c": graph.nodes[1][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                    }
                )
            }
        )
        # check edges
        assert (0, ref._node_id, "a.b") in graph.edges
        assert (1, ref._node_id, "a.c") in graph.edges

        ref = builder.collect(
            {
                "a": [
                    ConcreteReference(0, graph, builder),
                    ConcreteReference(1, graph, builder),
                ]
            }
        )
        # check node attributes
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ].config.lookup == {"a": ["a.0", "a.1"]}
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.COLLECT
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.construct(
            {
                "a.0": graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                "a.1": graph.nodes[1][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
            }
        )
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MappingType.construct(
            {
                "a": SequenceType(
                    graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE], length=2
                )
            }
        )
        # check edges
        assert (0, ref._node_id, "a.0") in graph.edges
        assert (1, ref._node_id, "a.1") in graph.edges

    def test_compute(self) -> None:
        builder = DataFlowGraphBuilder()
        ref = builder.source(MappingType.construct({"x": MockType}))

        # mock collect operation
        builder.collect = MagicMock(return_value=builder.const(True, MockType))

        # create a mock node instance
        mock_node = MagicMock(spec=BaseDataAugmentor)

        with (
            patch("hyped.core.builder.DataFlowGraphBuilder._add_node_to_graph") as mock_add_node,
            patch("hyped.core.builder.FeatureEngine") as mock_feature_engine_type,
        ):
            # mock the feature engine
            mock_feature_engine_type.return_value.__enter__.return_value = (
                mock_feature_engine_type.return_value
            )
            mock_feature_engine_type.return_value.get_features_and_objects.side_effect = (
                lambda x, y: ({"x": x}, {"y": y}, {"y": MockType})
            )

            # call the builder
            builder.compute(mock_node, {"x": ref, "y": False})

            # make sure the feature engine was called
            mock_feature_engine_type.return_value.validate_signature.assert_called_once()
            mock_feature_engine_type.return_value.validate_arguments.assert_called_once_with(
                x=build_feature_from_reference(ref), y=False
            )
            # make sure all objects where collected
            builder.collect.assert_called_once_with(False, MockType)
            # make sure the output feature was inferred correctly
            mock_feature_engine_type.return_value.build_return_feature.assert_called_once_with(
                {
                    "x": build_feature_from_reference(ref),
                    "y": build_feature_from_reference(builder.collect.return_value),
                }
            )
            # make sure the compute node was added to the graph
            mock_add_node.assert_called_once_with(
                node_obj=mock_node,
                node_type=DataFlowGraph.NodeType.DATA_AUGMENTOR,
                inputs={"x": ref, "y": builder.collect.return_value},
                output_dtype=mock_feature_engine_type.return_value.build_return_feature.return_value.dtype,
                node_id=None,
            )
