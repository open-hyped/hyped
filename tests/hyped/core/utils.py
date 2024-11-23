from typing import Hashable
from unittest.mock import MagicMock

import networkx as nx

from hyped.core.features.types import BoolType as MockType
from hyped.core.features.types import Type
from hyped.core.graph import DataFlowGraph, _build_dependency_graph


def build_graph_from_edge_list(
    edges: list[tuple[Hashable, Hashable]],
    node_types: None | dict[Hashable, DataFlowGraph.NodeType] = None,
    stop_at_node: None | Hashable = None,
    output_type: Type = MockType,
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
            node_obj=MagicMock(),
            node_type=node_types[node],
            inputs={str(u): refs[u] for u, _ in tmp_graph.in_edges(node)},
            output_type=output_type,
            node_id=node,
        )

    assert nx.is_isomorphic(tmp_graph, graph), "Error building graph from edges"

    return graph
