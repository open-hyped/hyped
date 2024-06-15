"""Module for Data Flow Graph Optimization.

This module provides an optimizer for data flow graphs, which applies various optimization
techniques to improve program efficiency and reduce redundant computations.

The optimizer module includes methods for optimizing data flow graphs, such as:

1. Prune Redundant Nodes: The optimizer prunes the data flow graph, removing nodes that
   do not contribute to producing the desired output.

2. Common Subexpression Elimination (CSE): Identifies and eliminates redundant computations
   by recognizing and reusing common subexpressions in the data flow graph.

3. Constant Folding/Propagation: Evaluates constant expressions at compile time and replaces
   them with their computed values to simplify the data flow graph.

The DataFlowGraphOptimizer class within this module provides these optimization methods,
which can be applied individually or in combination to optimize a given data flow graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

from hyped.common.feature_key import FeatureKey
from hyped.data.flow.core.nodes.const import Const
from hyped.data.flow.core.refs.inputs import InputRefs
from hyped.data.flow.core.refs.ref import FeatureRef
from hyped.data.flow.processors.ops.collect import CollectFeatures

from .graph import DataFlowGraph


class DataFlowGraphOptimizer(object):
    """Optimizer for Data Flow Graphs.

    This optimizer applies several optimization techniques to the given data flow graph
    in order to improve its efficiency and reduce redundant computations. The optimization
    steps include:

    1. Prune Unnecessary Nodes
    2. Common Subexpression Elimination (CSE)
    3. Constant Folding/Propagation
    """

    def cse(
        self, graph: DataFlowGraph
    ) -> tuple[DataFlowGraph, dict[int, int]]:
        """Performs Common Subexpression Elimination (CSE) on the data flow graph.

        This method performs Common Subexpression Elimination (CSE) on the given data flow graph.
        It optimizes the graph by identifying and eliminating redundant computations.

        Consider the following data flow graph:

        .. code-block:: python

            x = a + b
            y = c * d
            z = a + b

        After applying Common Subexpression Elimination (CSE), the redundant computation
        :code:`a + b` is eliminated, resulting in the following optimized graph:

        .. code-block:: python

            x = a + b
            y = c * d
            z = x

        The node IDs before and after optimization are mapped as follows:
        :code:`{0: 0, 1: 1, 2: 0}`

        Args:
            graph (DataFlowGraph): The data flow graph.

        Returns:
            tuple[DataFlowGraph, dict[int, int]]: The optimized data flow graph and a mapping
                of node IDs before and after optimization.
        """

        @dataclass
        class _CSE_NodeIdentifier(object):
            """Helper class to identify redundant nodes."""

            node_type: DataFlowGraph.NodeType
            node_config: str
            in_edge_identifiers: list[tuple[int, str, FeatureKey]]
            node_id: int = field(default=-1, compare=False)

        cse_graph = DataFlowGraph()
        node_id_mapping: dict[int, int] = {}

        key = lambda n: graph.nodes[n][DataFlowGraph.NodeProperty.DEPTH]
        for _, layer in groupby(sorted(graph, key=key), key=key):
            cse_layer = []

            for node_id in layer:
                # build identifiers for incoming edges with
                # source nodes mapped to nodes in optimized graph
                in_edge_identifiers = [
                    (
                        node_id_mapping[src_node_id],
                        edge_data[DataFlowGraph.EdgeProperty.NAME],
                        edge_data[DataFlowGraph.EdgeProperty.KEY],
                    )
                    for src_node_id, _, edge_data in graph.in_edges(
                        node_id, data=True
                    )
                ]

                node_data = graph.nodes[node_id]
                obj = node_data[DataFlowGraph.NodeProperty.NODE_OBJ]
                # create cse node identifier
                identifier = _CSE_NodeIdentifier(
                    node_type=node_data[DataFlowGraph.NodeProperty.NODE_TYPE],
                    node_config=getattr(obj, "config", None),
                    in_edge_identifiers=in_edge_identifiers,
                )

                if identifier in cse_layer:
                    # add entry to mapping
                    cse_node = cse_layer[cse_layer.index(identifier)]
                    node_id_mapping[node_id] = cse_node.node_id

                else:
                    if obj is None:
                        # add source node to optimized graph
                        features = node_data[
                            DataFlowGraph.NodeProperty.OUT_FEATURES
                        ]
                        identifier.node_id = cse_graph.add_source_node(
                            features
                        )

                    elif isinstance(obj, Const):
                        # add the constant to the cse graph
                        features = node_data[
                            DataFlowGraph.NodeProperty.OUT_FEATURES
                        ]
                        identifier.node_id = cse_graph.add_processor_node(
                            obj, None, features
                        )

                    elif isinstance(obj, CollectFeatures):
                        map_node_id = lambda r: (
                            r.model_copy(
                                update=dict(
                                    node_id_=node_id_mapping[r.node_id_],
                                    flow_=cse_graph,
                                )
                            )
                            if isinstance(r, FeatureRef)
                            else r
                        )
                        collection = obj.collection._map(map_node_id)
                        identifier.node_id = obj.call(
                            collection=collection, flow=cse_graph
                        ).node_id_

                    else:
                        # build call keyword arguments
                        named_refs = {
                            name: cse_graph.get_node_output_ref(src_node_id)[
                                key
                            ]
                            for src_node_id, name, key in in_edge_identifiers
                        }
                        # call processor
                        identifier.node_id = obj.call(**named_refs).node_id_

                    # update cse layer and node id mapping
                    cse_layer.append(identifier)
                    node_id_mapping[node_id] = identifier.node_id

        return cse_graph, node_id_mapping

    def constant_folding(self, graph: DataFlowGraph) -> DataFlowGraph:
        """Performs constant folding optimization on the data flow graph.

        Constant folding is an optimization technique used to evaluate constant expressions
        at compile time and replace them with their computed values. This method traverses
        the data flow graph and identifies expressions involving constants that can be
        evaluated statically. It then replaces these expressions with their computed values,
        eliminating redundant computations and simplifying the graph.

        Consider the following example:

        .. code-block:: python

            x = 10
            y = 5
            z = x + (-y)

        After applying constant folding optimization, the expression :code:`x + (-y)` is
        evaluated to :code:`x - y`, resulting in the following optimized graph:

        .. code-block:: python

            x = 10
            y = 5
            z = x - y

        Args:
            graph (DataFlowGraph): The data flow graph to be optimized.

        Returns:
            DataFlowGraph: The optimized data flow graph after constant folding.
        """
        # TODO
        return graph

    def optimize(
        self, graph: DataFlowGraph, leaf_nodes: set[int]
    ) -> tuple[DataFlowGraph, dict[int, int]]:
        """Optimizes the data flow graph.

        Args:
            graph (DataFlowGraph): The data flow graph.
            leaf_nodes (set[int]): Set of leaf node IDs.

        Returns:
            tuple[DataFlowGraph, dict[int, int]]: The optimized data flow graph and a mapping
                of node IDs before and after optimization.
        """
        # build dependency graph for the given set of nodes
        graph = graph.dependency_graph(leaf_nodes)

        # apply common sub-expresison elimination
        graph, leaf_nodes_mapping = self.cse(graph)

        # apply constant folding/propagation
        graph = self.constant_folding(graph)

        return graph, leaf_nodes_mapping
