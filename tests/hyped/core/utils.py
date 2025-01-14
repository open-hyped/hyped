from typing import Any, Hashable
from unittest.mock import MagicMock

import networkx as nx
import numpy as np

from hyped.core.builder import DataFlowGraphBuilder
from hyped.core.features.dtypes import BoolType as MockType
from hyped.core.features.dtypes import DType
from hyped.core.graph import DataFlowGraph, _build_dependency_graph


def build_graph(
    edges: (list[tuple[Hashable, Hashable]] | list[tuple[Hashable, Hashable, Hashable]]),
    node_types: None | dict[Hashable, DataFlowGraph.NodeType] = None,
    node_objects: dict[Hashable, Any] = {},
    stop_at_node: None | Hashable = None,
    output_type: DType = MockType,
) -> DataFlowGraph:
    """Helper function to construct a :class:`DataFlowGraph` from edges.

    This function builds a directed graph using the given edges, assigns node types, and
    incorporates specified node objects. It supports multi-directed graphs by allowing edges
    to have unique keys. A subgraph can be constructed by specifying a stopping node,
    limiting the graph traversal to nodes reachable up to that point.

    Args:
        edges (list[tuple[Hashable, Hashable]] | list[tuple[Hashable, Hashable, Hashable]]):
            The list of edges representing connections between nodes in the graph.
            Edges can be:
            - Two-tuples :code:`(u, v)`, where :code`u` is the source node and :code:`v` is
              the target node.
            - Three-tuples :code:`(u, v, key)`, where :code:`key` is a unique identifier for
              the edge in a multi-directed graph.
        node_types (None | dict[Hashable, DataFlowGraph.NodeType], optional):
            A mapping of node identifiers to their types in the :class:`DataFlowGraph`. If
            :code:`None`, the function will assign :class:`DataFlowGraph.NodeType.DATA_PROCESSOR`
            to all nodes by default, except for a single source node.
        node_objects (dict[Hashable, Any], optional):
            A dictionary mapping node identifiers to arbitrary objects representing node
            attributes. If a node is not in this dictionary, it will be assigned a default
            mock object (:code:`MagicMock()`).
        stop_at_node (None | Hashable, optional):
            A node identifier to limit graph construction to only nodes reachable
            up to (but excluding) this node.
        output_type (DType, optional):
            The output type of all nodes in the graph. Defaults to :code:`MockType`.

    Returns:
        DataFlowGraph:
            The constructed `DataFlowGraph` instance.

    Raises:
        AssertionError:
            If the graph contains more than one source node or if the constructed graph
            is not isomorphic to the input edge list.

    Notes:
        - Disconnected nodes (nodes not connected via edges) specified in :code:`node_types`
          will be added to the graph.
        - If :code:`node_types` is not provided, the function assumes all nodes are
          :class:`DataFlowGraph.NodeType.DATA_PROCESSOR`, except for a single source node,
          which is determined as the node with zero in-degree.
        - The function verifies that the constructed graph is isomorphic to the input edge
          representation.

    Example:

    .. code-block:: python

        edges = [
            ("A", "B"),
            ("B", "C", "edge_1"),
            ("C", "D"),
        ]
        node_types = {
            "A": DataFlowGraph.NodeType.SOURCE,
            "B": DataFlowGraph.NodeType.DATA_PROCESSOR,
            "C": DataFlowGraph.NodeType.DATA_PROCESSOR,
            "D": DataFlowGraph.NodeType.DATA_PROCESSOR,
        }
        graph = build_graph(edges, node_types)

    """

    if len(edges) > 0 and len(edges[0]) == 2:
        edges = [(u, v, str(u)) for u, v in edges]

    # build a temporary graph from the edges
    tmp_graph = nx.MultiDiGraph()
    tmp_graph.add_edges_from(edges)

    # add disconnected nodes to the graph from the node objects
    if node_types is not None:
        tmp_graph.add_nodes_from([n for n in node_types.keys() if n not in tmp_graph])

    # take only the subgraph up to a specified node
    if stop_at_node is not None:
        tmp_graph = nx.MultiDiGraph(_build_dependency_graph(tmp_graph, {stop_at_node}))
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
    builder = DataFlowGraphBuilder()

    refs = {}
    # add all the nodes from the tmp graph
    for node in nx.topological_sort(tmp_graph):
        if node_types[node] == DataFlowGraph.NodeType.SOURCE:
            refs[node] = builder.source(output_type, node_id=node)

        else:
            refs[node] = builder._add_node_to_graph(
                node_obj=node_objects.get(node, MagicMock()),
                node_type=node_types[node],
                inputs={k: refs[u] for u, _, k in tmp_graph.in_edges(node, keys=True)},
                output_dtype=output_type,
                node_id=node,
            )

    return builder.graph


class NumpyArrayMatcher:
    def __init__(self, expected_array):
        self.expected_array = expected_array

    def __eq__(self, other):
        return np.array_equal(self.expected_array, other)
