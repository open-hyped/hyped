from itertools import chain
from typing import Hashable
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from hyped.core.features.reference import Reference
from hyped.core.features.types import BoolType as MockType
from hyped.core.features.types import MappingType, SequenceType
from hyped.core.graph import DataFlowGraph, _build_dependency_graph, _compute_node_depth
from hyped.core.nodes.aggregator import BaseDataAggregator
from hyped.core.nodes.augmenter import BaseDataAugmenter
from hyped.core.nodes.processor import BaseDataProcessor
from hyped.core.typing import PartitionId


@pytest.mark.parametrize(
    "edges, expected_depths, raises_error",
    [
        # Simple linear graph (chain)
        ([(0, 1), (1, 2), (2, 3)], {0: 0, 1: 1, 2: 2, 3: 3}, False),
        # Tree graph
        ([(0, 1), (0, 2), (1, 3), (1, 4)], {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}, False),
        # Complex DAG
        ([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}, False),
        # Graph with a cycle (should raise ValueError)
        ([(0, 1), (1, 2), (2, 0)], None, True),
        # Disconnected graph (each component is considered separately)
        ([(0, 1), (2, 3)], {0: 0, 1: 1, 2: 0, 3: 1}, False),
    ],
)
def test_compute_node_depth(
    edges: list[tuple[Hashable, Hashable]], expected_depths: dict[Hashable, int], raises_error: bool
) -> None:
    G = nx.DiGraph()
    G.add_edges_from(edges)

    if raises_error:
        with pytest.raises(ValueError):
            _compute_node_depth(G)
    else:
        result = _compute_node_depth(G)
        assert result == expected_depths


@pytest.mark.parametrize(
    "edges, nodes, stop_nodes, expected_edges, raises_error",
    [
        # Simple graph with no stop nodes
        ([(0, 1), (1, 2), (2, 3)], {3}, set(), [(0, 1), (1, 2), (2, 3)], False),
        # Graph with a stop node
        ([(0, 1), (1, 2), (2, 3)], {3}, {1}, [(1, 2), (2, 3)], False),
        # Graph with multiple target nodes and no stop nodes
        (
            [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5)],
            {3, 5},
            set(),
            [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5)],
            False,
        ),
        # Graph with stop nodes cutting off one branch
        (
            [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)],
            {3, 5},
            {1},
            [(1, 2), (2, 3), (1, 4), (4, 5)],
            False,
        ),
        # Graph with stop nodes excluding all dependencies
        ([(0, 1), (1, 2), (2, 3)], {3}, {2}, [(2, 3)], False),
        # Node not in the graph (should raise an error)
        ([(0, 1), (1, 2)], {3}, set(), None, True),
    ],
)
def test_build_dependency_graph(
    edges: list[tuple[Hashable, Hashable]],
    nodes: set[Hashable],
    stop_nodes: set[Hashable],
    expected_edges: list[tuple[Hashable, Hashable]],
    raises_error: bool,
) -> None:
    G = nx.DiGraph()
    G.add_edges_from(edges)

    if raises_error:
        with pytest.raises(AssertionError):
            _build_dependency_graph(G, nodes, stop_nodes)
    else:
        subgraph = _build_dependency_graph(G, nodes, stop_nodes)
        # Check that the edges in the resulting subgraph match the expected edges
        assert set(subgraph.edges) == set(expected_edges)
        # Check that all nodes in the subgraph are reachable from the target nodes
        reachable_nodes = (nx.dfs_preorder_nodes(subgraph.reverse(), source=node) for node in nodes)
        reachable_nodes = set(list(chain.from_iterable(reachable_nodes)))
        assert all(node in reachable_nodes for node in subgraph.nodes)


class TestDataFlowGraph:
    def build_graph_from_edge_list(
        self,
        edges: list[tuple[Hashable, Hashable]],
        node_types: None | dict[Hashable, DataFlowGraph.NodeType] = None,
        stop_at_node: None | Hashable = None,
    ) -> DataFlowGraph:
        # build a temporary graph from the edges
        tmp_graph = nx.MultiDiGraph()
        tmp_graph.add_edges_from(edges)

        if stop_at_node is not None:
            tmp_graph = nx.DiGraph(_build_dependency_graph(tmp_graph, {stop_at_node}))
            tmp_graph.remove_node(stop_at_node)

        if node_types is None:
            # by default all nodes are data processors, except for the source node
            node_types = {n: DataFlowGraph.NodeType.DATA_PROCESSOR for n in tmp_graph.nodes}
            # find the source node
            source_nodes = [node for node, in_degree in tmp_graph.in_degree() if in_degree == 0]
            assert len(source_nodes) == 1, f"Expected one source node, got {len(source_nodes)}"
            # set the node type
            node_types[source_nodes[0]] = DataFlowGraph.NodeType.SOURCE

        # build the data flow graph
        graph = DataFlowGraph()

        refs = {}

        for node in nx.topological_sort(tmp_graph):
            refs[node] = graph.add_node(
                node_obj=node,
                node_type=node_types[node],
                inputs={str(u): refs[u] for u, _ in tmp_graph.in_edges(node)},
                output_type=MockType,
                node_id=node,
            )

        assert nx.is_isomorphic(tmp_graph, graph), "Error building graph from edges"

        return graph

    @pytest.mark.parametrize(
        "edges, expected_depth, raises_error",
        [
            # Simple linear graph (chain)
            ([(0, 1), (1, 2), (2, 3)], 4, False),
            # Tree graph
            ([(0, 1), (0, 2), (1, 3), (1, 4)], 3, False),
            # Complex DAG
            ([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], 4, False),
        ],
    )
    def test_depth_property(
        self, edges: list[tuple[Hashable, Hashable]], expected_depth: int, raises_error: bool
    ) -> None:
        G = self.build_graph_from_edge_list(edges)

        if raises_error:
            with pytest.raises(ValueError):
                _ = G.depth
        else:
            assert G.depth == expected_depth

    @pytest.mark.parametrize(
        "edges, expected_width, raises_error",
        [
            # Simple linear graph (chain)
            ([(0, 1), (1, 2), (2, 3)], 1, False),
            # Tree graph (maximum width is at the second layer)
            ([(0, 1), (0, 2), (1, 3), (1, 4)], 2, False),
            # Complex DAG
            ([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], 2, False),
            # Balanced binary tree with height 2
            ([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], 4, False),
        ],
    )
    def test_width_property(
        self, edges: list[tuple[Hashable, Hashable]], expected_width: int, raises_error: bool
    ) -> None:
        G = self.build_graph_from_edge_list(edges)

        if raises_error:
            with pytest.raises(ValueError):
                _ = G.width
        else:
            assert G.width == expected_width

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
            # Linear graph including augmenter and aggregator nodes
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    3: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                },
                [
                    (DataFlowGraph.Partition.DEFAULT.value, 2),
                    (2, DataFlowGraph.Partition.AGGREGATED),
                ],
            ),
            # Linear graph with multiple augmenters
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    3: DataFlowGraph.NodeType.DATA_AUGMENTER,
                },
                [(DataFlowGraph.Partition.DEFAULT.value, 1), (1, 2), (2, 3)],
            ),
            # Tree with two branches
            (
                [(0, 1), (0, 2), (1, 3), (2, 4)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTER,
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
        graph = self.build_graph_from_edge_list(edges, nodes)
        partition = graph.build_partition_graph()

        target_partition_graph = nx.DiGraph()
        target_partition_graph.add_nodes_from(
            [
                DataFlowGraph.Partition.CONST.value,
                DataFlowGraph.Partition.DEFAULT.value,
            ]
        )

        target_partition_graph.add_edges_from(partition_edges)

        assert nx.utils.misc.graphs_equal(partition, target_partition_graph)

    @pytest.mark.parametrize(
        "edges, in_nodes, expected_depth",
        [
            # Add unconnected node (const node for example)
            ([(0, 1), (1, 2), (2, 3)], [], 0),
            # Add single connection node at layer 2
            ([(0, 1), (1, 2), (2, 3)], [1], 2),
            # Add single connection node at layer 3
            ([(0, 1), (1, 2), (2, 3)], [2], 3),
            # Add single connection node at layer 4
            ([(0, 1), (1, 2), (2, 3)], [3], 4),
            # Add multi-connected node at layer 3
            ([(0, 1), (1, 2), (2, 3)], [0, 1, 2], 3),
        ],
    )
    def test_add_node(
        self, edges: list[tuple[Hashable, Hashable]], in_nodes: tuple[Hashable], expected_depth: int
    ) -> None:
        graph = self.build_graph_from_edge_list(edges)

        node_obj = MagicMock()
        node_type = (
            DataFlowGraph.NodeType.DATA_PROCESSOR
            if len(in_nodes) > 0
            else DataFlowGraph.NodeType.CONST
        )

        ref = graph.add_node(
            node_obj=node_obj,
            node_type=node_type,
            inputs={str(u): Reference(_node_id=u, _graph=graph) for u in in_nodes},
            output_type=MockType,
        )

        attrs = graph.nodes[ref._node_id]
        # check expected node attributes
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ] == node_obj
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == node_type
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MockType
        assert attrs[DataFlowGraph.NodeAttribute.DEPTH] == expected_depth
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.from_dict(
            {str(u): MockType for u in in_nodes}
        )

        # check input edges
        for u in in_nodes:
            assert (u, ref._node_id, str(u)) in graph.edges

    def test_add_source_node(self) -> None:
        # create a data flow graph
        graph = DataFlowGraph()
        ref = graph.add_source_node(MockType)
        # make sure the source node was added to the graph
        assert graph.src_node_id in graph
        assert graph.src_node_id == ref._node_id
        assert graph.src_dtype == MockType

        # cannot add another source node
        with pytest.raises(RuntimeError):
            graph.add_source_node(MockType)

        # new graph containing source node keeps the source node
        new_graph = DataFlowGraph(graph)
        assert new_graph.src_node_id is not None
        assert new_graph.src_node_id in new_graph
        assert new_graph.src_node_id == graph.src_node_id
        assert new_graph.src_dtype == MockType

        # sub-graph not containing source node resets source node
        sub_graph = DataFlowGraph(graph.subgraph([]))
        assert sub_graph.src_node_id is None

    @patch("hyped.core.graph.pa.array")
    def test_add_const_node(self, mock_pa_array: MagicMock) -> None:
        graph = DataFlowGraph()

        value = MagicMock()
        # add the node to the graph
        ref = graph.add_const_node(value, MockType)

        # make sure node was added as expected
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ] == mock_pa_array()
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.CONST
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MockType
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.from_dict({})

    def test_add_collect_node(self) -> None:
        # create simple linear graph
        graph = self.build_graph_from_edge_list([(0, 1), (1, 2), (2, 3)])

        ref = graph.add_collect_node(
            {
                "a": Reference(_node_id=0, _graph=graph),
                "b": Reference(_node_id=1, _graph=graph),
            }
        )
        # check node attributes
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ].config.lookup == {"a": "a", "b": "b"}
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.COLLECT
        assert (
            attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE]
            == attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
            == MappingType.from_dict(
                {
                    "a": graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                    "b": graph.nodes[1][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                }
            )
        )
        # check edges
        assert (0, ref._node_id, "a") in graph.edges
        assert (1, ref._node_id, "b") in graph.edges

        ref = graph.add_collect_node(
            {
                "a": {
                    "b": Reference(_node_id=0, _graph=graph),
                    "c": Reference(_node_id=1, _graph=graph),
                },
            }
        )
        # check node attributes
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ].config.lookup == {
            "a": {"b": "a.b", "c": "a.c"}
        }
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.COLLECT
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.from_dict(
            {
                "a.b": graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                "a.c": graph.nodes[1][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
            }
        )
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MappingType.from_dict(
            {
                "a": MappingType.from_dict(
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

        ref = graph.add_collect_node(
            {"a": [Reference(_node_id=0, _graph=graph), Reference(_node_id=1, _graph=graph)]}
        )
        # check node attributes
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ].config.lookup == {"a": ["a.0", "a.1"]}
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == DataFlowGraph.NodeType.COLLECT
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.from_dict(
            {
                "a.0": graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
                "a.1": graph.nodes[1][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE],
            }
        )
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MappingType.from_dict(
            {
                "a": SequenceType(
                    graph.nodes[0][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE], length=2
                )
            }
        )
        # check edges
        assert (0, ref._node_id, "a.0") in graph.edges
        assert (1, ref._node_id, "a.1") in graph.edges

    @pytest.mark.parametrize(
        "node_cls, expected_node_type",
        [
            (BaseDataProcessor, DataFlowGraph.NodeType.DATA_PROCESSOR),
            (BaseDataAugmenter, DataFlowGraph.NodeType.DATA_AUGMENTER),
            (BaseDataAggregator, DataFlowGraph.NodeType.DATA_AGGREGATOR),
        ],
    )
    @patch("hyped.core.graph.FeatureEngine")
    def test_add_compute_node(
        self,
        mock_feature_engine: MagicMock,
        node_cls: type,
        expected_node_type: DataFlowGraph.NodeType,
    ) -> None:
        # create a graph with a source node
        graph = DataFlowGraph()
        src_ref = graph.add_source_node(MockType)
        # add a node of the specified type to the graph
        node = MagicMock(__class__=node_cls)
        ref = graph.add_compute_node(node, {"x": src_ref})
        # check the added node type
        attrs = graph.nodes[ref._node_id]
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == expected_node_type

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
            # Augmenters are not part of their own partition
            (
                [(0, 1), (1, 2)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                1,
                DataFlowGraph.Partition.DEFAULT,
                False,
            ),
            # Augmenters introduce a new partition
            (
                [(0, 1), (1, 2)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    2: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                2,
                1,  # partition uses the same id as the augmenter node
                False,
            ),
            # Chaining augmenters the latest augmenter partition wins
            (
                [(0, 1), (1, 2), (2, 3), (1, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    3: DataFlowGraph.NodeType.DATA_AUGMENTER,
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
                    1: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    2: DataFlowGraph.NodeType.DATA_AUGMENTER,
                    3: DataFlowGraph.NodeType.DATA_AUGMENTER,
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
        # build the data flow
        graph = self.build_graph_from_edge_list(edges, node_types, node_id)

        # get the node type and build the reference instances
        node_type = node_types[node_id]
        refs = [Reference(_node_id=u, _graph=graph) for u, v in edges if v == node_id]

        if raises_error:
            with pytest.raises(RuntimeError):
                graph.infer_node_partition(node_type, refs)
        else:
            partition = graph.infer_node_partition(node_type, refs)
            assert partition == expected_partition

    def test_get_dtype_from_reference(self) -> None:
        graph = self.build_graph_from_edge_list([(0, 1), (1, 2)])

        # works fine
        graph.get_dtype_from_reference(Reference(_node_id=0, _graph=graph))
        # wrong graph
        with pytest.raises(RuntimeError):
            graph.get_dtype_from_reference(Reference(_node_id=0, _graph=MagicMock()))
        # wrong node id
        with pytest.raises(RuntimeError):
            graph.get_dtype_from_reference(Reference(_node_id="INVALID", _graph=graph))

    def test_get_feature_from_reference(self) -> None:
        graph = self.build_graph_from_edge_list([(0, 1), (1, 2)])

        # works fine
        graph.get_feature_from_reference(Reference(_node_id=0, _graph=graph))
        # wrong graph
        with pytest.raises(RuntimeError):
            graph.get_feature_from_reference(Reference(_node_id=0, _graph=MagicMock()))
        # wrong node id
        with pytest.raises(RuntimeError):
            graph.get_feature_from_reference(Reference(_node_id="INVALID", _graph=graph))

    @pytest.mark.parametrize(
        "edges, subgraph_nodes, expected_edges",
        [
            ([(0, 1), (1, 2), (2, 3)], [1, 2], [(0, 1)]),
            ([(0, 1), (1, 2), (2, 3)], [2, 3], [(1, 2)]),
            ([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 5)], [4, 5], [(1, 4), (2, 4), (3, 5)]),
        ],
    )
    def test_subgraph_in_edges(
        self,
        edges: list[tuple[Hashable, Hashable]],
        subgraph_nodes: list[Hashable],
        expected_edges: list[tuple[Hashable, Hashable]],
    ) -> None:
        # build graph and get the subgraph edges
        graph = self.build_graph_from_edge_list(edges)
        subgraph = graph.subgraph(subgraph_nodes)
        edges = graph.subgraph_in_edges(subgraph)
        # check edges
        assert set(expected_edges) == set([(u, v) for u, v, _ in edges])

    @pytest.mark.parametrize(
        "edges, subgraph_nodes, expected_edges",
        [
            ([(0, 1), (1, 2), (2, 3)], [1, 2], [(2, 3)]),
            ([(0, 1), (1, 2), (2, 3)], [0, 1], [(1, 2)]),
            ([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 5)], [1, 2, 3], [(1, 4), (2, 4), (3, 5)]),
        ],
    )
    def test_subgraph_out_edges(
        self,
        edges: list[tuple[Hashable, Hashable]],
        subgraph_nodes: list[Hashable],
        expected_edges: list[tuple[Hashable, Hashable]],
    ) -> None:
        # build graph and get the subgraph edges
        graph = self.build_graph_from_edge_list(edges)
        subgraph = graph.subgraph(subgraph_nodes)
        edges = graph.subgraph_out_edges(subgraph)
        # check edges
        assert set(expected_edges) == set([(u, v) for u, v, _ in edges])

    @pytest.mark.parametrize(
        "edges, node_types, partition, expected_nodes",
        [
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                DataFlowGraph.Partition.DEFAULT,
                [0, 1, 2],
            ),
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                DataFlowGraph.Partition.AGGREGATED,
                [3],
            ),
        ],
    )
    def test_get_partition(
        self,
        edges: list[tuple[Hashable, Hashable]],
        node_types: dict[Hashable, DataFlowGraph.NodeType],
        partition: PartitionId,
        expected_nodes: list[Hashable],
    ) -> None:
        graph = self.build_graph_from_edge_list(edges, node_types)
        subgraph = graph.get_partition(partition)
        assert set(list(subgraph.nodes)) == set(expected_nodes)

    @pytest.mark.parametrize(
        "edges, node_types, partition, expected_nodes",
        [
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                DataFlowGraph.Partition.DEFAULT,
                [3],
            ),
            (
                [(0, 1), (1, 2), (2, 3)],
                {
                    0: DataFlowGraph.NodeType.SOURCE,
                    1: DataFlowGraph.NodeType.DATA_PROCESSOR,
                    2: DataFlowGraph.NodeType.DATA_AGGREGATOR,
                    3: DataFlowGraph.NodeType.DATA_PROCESSOR,
                },
                DataFlowGraph.Partition.AGGREGATED,
                [0, 1, 2],
            ),
        ],
    )
    def test_drop_partition(
        self,
        edges: list[tuple[Hashable, Hashable]],
        node_types: dict[Hashable, DataFlowGraph.NodeType],
        partition: PartitionId,
        expected_nodes: list[Hashable],
    ) -> None:
        graph = self.build_graph_from_edge_list(edges, node_types)
        subgraph = graph.drop_partition(partition)
        assert set(list(subgraph.nodes)) == set(expected_nodes)
