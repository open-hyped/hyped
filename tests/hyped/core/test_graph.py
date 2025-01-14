from itertools import chain
from typing import Hashable
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from hyped.core.features.dtypes import BoolType as MockType
from hyped.core.features.dtypes import MappingType
from hyped.core.graph import (
    DataFlowGraph,
    _build_dependency_graph,
    _compute_node_depth,
    random_uuid,
)
from hyped.core.nodes.processor import BaseDataProcessor
from hyped.core.typing import PartitionId

from .utils import build_graph


def test_random_uuid() -> None:
    assert len(set([str(random_uuid()) for _ in range(1_000_000)])) == 1_000_000


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
        G = build_graph(edges)

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
        G = build_graph(edges)

        if raises_error:
            with pytest.raises(ValueError):
                _ = G.width
        else:
            assert G.width == expected_width

    def test_add_source_node(self) -> None:
        # create a data flow graph
        graph = DataFlowGraph()
        src_node_id = graph.add_source_node(MockType, "SOURCE_NODE_ID")
        # make sure the source node was added to the graph
        assert graph.src_node_id is not None
        assert graph.src_node_id in graph
        assert graph.src_node_id in src_node_id
        assert graph.src_dtype == MockType

        # cannot add another source node
        with pytest.raises(RuntimeError):
            graph.add_source_node(MockType, "NEW_SOURCE_NODE_ID")

        # new graph containing source node keeps the source node
        new_graph = DataFlowGraph(graph)
        assert new_graph.src_node_id is not None
        assert new_graph.src_node_id in new_graph
        assert new_graph.src_node_id == graph.src_node_id
        assert new_graph.src_dtype == MockType

        # sub-graph not containing source node resets source node
        sub_graph = DataFlowGraph(graph.subgraph([]))
        assert sub_graph.src_node_id is None

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
        graph = build_graph(edges)

        node_obj = MagicMock()
        node_type = (
            DataFlowGraph.NodeType.DATA_PROCESSOR
            if len(in_nodes) > 0
            else DataFlowGraph.NodeType.CONST
        )

        node_id = graph.add_node(
            node_obj=node_obj,
            node_type=node_type,
            inputs={str(u): u for u in in_nodes},
            output_dtype=MockType,
            partition=DataFlowGraph.Partition.DEFAULT,
            out_partition=DataFlowGraph.Partition.DEFAULT,
            node_id="NODE_ID",
        )

        attrs = graph.nodes[node_id]
        # check expected node attributes
        assert attrs[DataFlowGraph.NodeAttribute.NODE_OBJ] == node_obj
        assert attrs[DataFlowGraph.NodeAttribute.NODE_TYPE] == node_type
        assert attrs[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] == MockType
        assert attrs[DataFlowGraph.NodeAttribute.DEPTH] == expected_depth
        assert attrs[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] == MappingType.construct(
            {str(u): MockType for u in in_nodes}
        )

        # check input edges
        for u in in_nodes:
            assert (u, node_id, str(u)) in graph.edges

    def test_get_output_dtype(self) -> None:
        graph = build_graph([(0, 1), (1, 2)])
        # works fine
        graph.get_output_dtype(0)
        # invalid node id
        with pytest.raises(RuntimeError):
            graph.get_output_dtype("INVALID")

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
        graph = build_graph(edges)
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
        graph = build_graph(edges)
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
        graph = build_graph(edges, node_types)
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
        graph = build_graph(edges, node_types)
        subgraph = graph.drop_partition(partition)
        assert set(list(subgraph.nodes)) == set(expected_nodes)

    @pytest.mark.parametrize(
        "graph",
        [
            build_graph([], {0: DataFlowGraph.NodeType.SOURCE}, {0: None}),
            build_graph(
                [(0, 1)],
                {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.DATA_PROCESSOR},
                {0: None, 1: MagicMock(__spec__=BaseDataProcessor)},
            ),
            build_graph(
                [(0, 1)],
                {0: DataFlowGraph.NodeType.SOURCE, 1: DataFlowGraph.NodeType.CAST},
                {0: None, 1: None},
            ),
        ],
    )
    def test_dict_serialization(self, graph: DataFlowGraph) -> None:
        mock_node_from_config_dict = {
            node.config.to_dict.return_value: node
            for _, node in graph.nodes(data=DataFlowGraph.NodeAttribute.NODE_OBJ)
            if isinstance(node, MagicMock)
        }.get

        with patch(
            "hyped.core.graph.AutoConfigurable.from_config_dict", mock_node_from_config_dict
        ):
            reconstructed_graph = DataFlowGraph.from_dict(graph.to_dict())

        assert nx.utils.misc.graphs_equal(reconstructed_graph, graph)
