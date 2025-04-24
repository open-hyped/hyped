"""Module for Data Flow Graph Optimization.

This module provides an optimizer for data flow graphs, which applies various optimization
techniques to improve program efficiency and reduce redundant computations.

The optimizer module includes methods for optimizing data flow graphs, such as:

1. **Prune Redundant Nodes**: The optimizer prunes the data flow graph, removing nodes that
   do not contribute to producing the desired output.

2. **Constant Expression Evaluation**: Pre-computes the constant partition of the graph, which
   consists solely of constant values and has no dependencies on other parts of the graph, and
   replaces these constants with their computed values.

3. **Common Subexpression Elimination (CSE)**: Identifies and eliminates redundant computations
   by recognizing and reusing common subexpressions in the data flow graph.

4. **Constant Folding/Propagation**: Evaluates constant expressions at compile time and replaces
   them with their computed values to simplify the data flow graph.

The DataFlowGraphOptimizer class within this module provides these optimization methods,
which can be applied individually or in combination to optimize a given data flow graph.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any

import networkx as nx
import pyarrow as pa

from .builder import DataFlowGraphBuilder
from .executor import DataFlowGraphExecutor
from .features.dtypes import BoolType, DType, MappingType
from .features.reference import ConcreteReference
from .graph import DataFlowGraph
from .ops.mapping import MappingGetItem
from .typing import NodeId, PartitionId


class DataFlowGraphOptimizer(object):
    """Optimizer for Data Flow Graphs.

    This optimizer applies several optimization techniques to the given data flow graph
    in order to improve its efficiency and reduce redundant computations. The optimization
    steps include:

    1. Prune Unnecessary Nodes
    2. Constant Expressions Evaluation
    3. Common Subexpression Elimination (CSE)
    4. Constant Folding/Propagation

    Note the difference between constant expression evaluation and constant folding. Constant
    expression evaluation focuses on evaluating the constant partition of the graph, which
    consists solely of constant values and has no dependencies on other parts of the graph.
    In contrast, constant folding aims to simplify expressions that include both constant
    and non-constant values by precomputing the constant parts of these expressions.
    """

    def cse(self, graph: DataFlowGraph) -> tuple[DataFlowGraph, dict[NodeId, NodeId]]:
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
            DataFlowGraph, dict[NodeId, NodeId]]: The optimized data flow graph and a mapping
                of node IDs before and after optimization.
        """

        @dataclass
        class NodeIdentifier(object):
            """Helper class to identify redundant nodes."""

            node_type: DataFlowGraph.NodeType
            node_config: Any
            in_edge_identifiers: list[tuple[ConcreteReference, str]]
            ref: ConcreteReference = field(default=None, compare=False)

        builder = DataFlowGraphBuilder()
        # maps nodes of the original graph to the nodes in the cse-graph
        # this is a non-injective function as multiple nodes in the original
        # graph can be mapped to the same target node during optimization
        node_mapping: dict[NodeId, NodeId] = {}

        # backlog of trace nodes that occured during processing but
        # were not added to the graph yet
        trace_nodes: dict[PartitionId, set[NodeId]] = defaultdict(set)
        partition_mapping: dict[PartitionId, PartitionId] = {}

        def key(n):
            return graph.nodes[n][DataFlowGraph.NodeAttribute.DEPTH]

        for _, layer in groupby(sorted(graph, key=key), key=key):
            cse_layer: list[NodeIdentifier] = []

            # separate trace nodes from non-trace nodes
            nt = {n: graph.nodes[n][DataFlowGraph.NodeAttribute.NODE_TYPE] for n in layer}
            layer_trace_nodes = [n for n, t in nt.items() if t == DataFlowGraph.NodeType.TRACE]
            layer_other_nodes = [n for n, t in nt.items() if t != DataFlowGraph.NodeType.TRACE]

            # capture trace nodes in backlog
            for trace_node in layer_trace_nodes:
                partition = graph.nodes[trace_node][DataFlowGraph.NodeAttribute.OUT_PARTITION]
                trace_nodes[partition].add(trace_node)

            # process all trace nodes which have defined target
            # partitions in the cse graph
            for tgt_partition in list(trace_nodes.keys()):
                if tgt_partition in partition_mapping.keys():
                    new_tgt_partition = partition_mapping[tgt_partition]

                    for trace_node_id in trace_nodes.pop(tgt_partition):
                        # trace nodes only have a single in-edge
                        src_node_id, _, k = next(iter(graph.in_edges(trace_node_id, keys=True)))
                        src_node_ref = node_mapping[src_node_id]

                        # create a node identifier for the trace node, use the new target
                        # partition of the trace node as an alias of its config
                        identifier = NodeIdentifier(
                            node_type=DataFlowGraph.NodeType.TRACE,
                            node_config=new_tgt_partition,
                            in_edge_identifiers=[(src_node_ref, k)],
                        )

                        if identifier in cse_layer:
                            identifier = cse_layer[cse_layer.index(identifier)]
                            node_mapping[trace_node_id] = identifier.ref

                        else:
                            identifier.ref = builder.trace(
                                ref=src_node_ref, tgt=new_tgt_partition, node_id=trace_node_id
                            )
                            cse_layer.append(identifier)
                            node_mapping[trace_node_id] = identifier.ref

            # process all other nodes
            for node_id in layer_other_nodes:
                # build identifiers for incoming edges with
                # source nodes mapped to nodes in optimized graph
                in_edge_identifiers = [
                    (node_mapping[src_node_id], k)
                    for src_node_id, _, k in graph.in_edges(node_id, keys=True)
                ]

                node_data = graph.nodes[node_id]
                node_type = node_data[DataFlowGraph.NodeAttribute.NODE_TYPE]
                node_obj = node_data[DataFlowGraph.NodeAttribute.NODE_OBJ]
                node_out_partition = node_data[DataFlowGraph.NodeAttribute.OUT_PARTITION]
                # create cse node identifier
                identifier = NodeIdentifier(
                    node_type=node_type,
                    node_config=getattr(node_obj, "config", node_obj),
                    in_edge_identifiers=in_edge_identifiers,
                )

                # do not add a new node if the node is already present in the layer
                if identifier in cse_layer:
                    identifier = cse_layer[cse_layer.index(identifier)]

                    assert identifier.ref is not None
                    node_mapping[node_id] = identifier.ref
                    partition_mapping[node_out_partition] = builder.graph.nodes[
                        identifier.ref._node_id
                    ][DataFlowGraph.NodeAttribute.OUT_PARTITION]

                else:
                    # read node feature properties
                    in_dtype = node_data[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE]
                    out_dtype = node_data[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]

                    if identifier.node_type == DataFlowGraph.NodeType.SOURCE:
                        identifier.ref = builder.source(out_dtype, node_id=node_id)

                    else:
                        inputs: dict[str, ConcreteReference] = {
                            name: node_mapping[src_node_ref._node_id]
                            for src_node_ref, name in in_edge_identifiers
                        }
                        identifier.ref = builder._add_node_to_graph(
                            node_obj=node_obj,
                            node_type=node_type,
                            inputs=inputs,
                            output_dtype=out_dtype,
                            node_id=node_id,
                        )

                    # make sure the input feature type
                    assert (
                        in_dtype
                        == builder.graph.nodes[identifier.ref._node_id][
                            DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE
                        ]
                    )

                    # update cse layer and node id mapping
                    assert identifier.ref is not None
                    cse_layer.append(identifier)
                    # the node keeps the same id
                    node_mapping[node_id] = identifier.ref
                    partition_mapping[node_out_partition] = builder.graph.nodes[
                        identifier.ref._node_id
                    ][DataFlowGraph.NodeAttribute.OUT_PARTITION]

        return builder.graph, {k: v._node_id for k, v in node_mapping.items()}

    def constant_evaluation(self, graph: DataFlowGraph, leaf_nodes: set[NodeId]) -> DataFlowGraph:
        """Pre-computes the constant partition of the data flow graph.

        This method evaluates the constant partition of the data flow graph,
        which is self-contained and has no outside dependencies. It creates a
        new graph from this partition, collects all outputs, and executes them
        using a data flow executor. The evaluated constants are then re-inserted
        into the main graph with their computed values.

        Consider the following example:

        .. code-block:: python

            x, y = 1, 2
            z = x + y

        After applying constant evaluation, the above is expression is precomputed to

        .. code-block:: python

            z = 3

        Args:
            graph (DataFlowGraph): The data flow graph to be optimized.
            leaf_nodes (set[NodeId]): The leaf nodes, ensured to be present in the optimized graph.

        Returns:
            DataFlowGraph: The optimized data flow graph with evaluated constants.
        """
        # get the constant partition of the graph and make sure
        # the sub-flow is self-contained, i.e. has no outside dependencies
        const_graph = graph.get_partition(DataFlowGraph.Partition.CONST)
        assert len(graph.subgraph_in_edges(const_graph)) == 0

        # identify all the trace nodes in the constant partition
        trace_node_ids = [
            node_id
            for node_id, node_type in const_graph.nodes(data=DataFlowGraph.NodeAttribute.NODE_TYPE)
            if node_type == DataFlowGraph.NodeType.TRACE
        ]
        # trace nodes are gateways between partitions, thus all
        # trace nodes in the constant partition should be leaf nodes
        assert all(const_graph.out_degree(node_id) == 0 for node_id in trace_node_ids)
        # drop the trace nodes
        const_graph = nx.restricted_view(const_graph, trace_node_ids, [])

        # check if there is anything to optimize in the constant partition
        # there are operations to collapse only if there are any edges within
        # the constant graph, otherwise the constant graph is either empty or
        # all nodes in the constant partition are source nodes which cannot
        # be optimized further
        if len(const_graph.edges) > 0:
            const_node_ids = list(const_graph.nodes)

            # create a dummy input feature for execution
            dummy_type = MappingType.construct({"field": BoolType})
            dummy_table = pa.table({"field": [True]}, schema=dummy_type.arrow_schema)

            # create a new graph from the view and apply common
            # sub-expression evaluation
            const_graph = DataFlowGraph(const_graph)
            const_graph, node_mapping = self.cse(const_graph)

            # map constant node ids and leaf nodes to potentially new values
            leaf_nodes = [
                node_mapping[node_id] if node_id in const_node_ids else node_id
                for node_id in leaf_nodes
            ]
            const_node_ids = [node_mapping[node_id] for node_id in const_node_ids]

            # create a builder instance and add a dummy source node
            const_builder = DataFlowGraphBuilder(const_graph)
            const_builder.source(dummy_type)

            # collect all outputs of all constant nodes in the graph
            collect = const_builder.collect(
                {
                    node_id: ConcreteReference(
                        _node_id=node_id, _graph=const_graph, _builder=const_builder
                    )
                    for node_id in const_node_ids
                }
            )

            # create an executor for the constant partition
            executor = DataFlowGraphExecutor(
                graph=const_graph, collect=collect, aggregation_manager=None
            )

            # execute the constant partition
            out = executor.run(dummy_table, index=[0], rank=0)
            # get usage of constants in the graph
            const_edges = graph.subgraph_out_edges(const_graph)

            # drop the constant partition in the original graph
            graph = DataFlowGraph(nx.restricted_view(graph, const_graph, []))
            builder = DataFlowGraphBuilder(graph)

            const_lookup: dict[NodeId, ConcreteReference] = {}
            # add constant leaf nodes
            for node_id in filter(const_graph.__contains__, leaf_nodes):
                # get the expected data type of the constant value
                dtype = const_graph.nodes[node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
                # add the constant node and track it
                ref = builder.const(out[node_id][0], dtype, node_id)
                const_lookup[node_id] = ref

            # add all required constants
            for const_node_id, tgt_node_id, key in const_edges:
                if const_node_id not in const_lookup:
                    # get the data type of the constant
                    dtype = const_graph.nodes[const_node_id][
                        DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE
                    ]
                    # add the constant node to the graph
                    ref = builder.const(out[const_node_id][0], dtype, const_node_id)
                    const_lookup[const_node_id] = ref

                # get the reference object from the lookup
                ref = const_lookup[const_node_id]
                # add the edge to the graph
                graph.add_edge(ref._node_id, tgt_node_id, key=key)

        return graph

    def constant_folding(self, graph: DataFlowGraph) -> DataFlowGraph:
        """Performs constant folding optimization on the data flow graph.

        Constant folding is an optimization technique used to evaluate constant expressions
        at compile time and replace them with their computed values. This method traverses
        the data flow graph and identifies expressions involving constants that can be
        evaluated statically. It then replaces these expressions with their computed values,
        eliminating redundant computations and simplifying the graph.

        Consider the following example:

        .. code-block:: python

            z = x + (-y)

        After applying constant folding optimization, the expression is simpified to

        .. code-block:: python

            z = x - y

        Args:
            graph (DataFlowGraph): The data flow graph to be optimized.

        Returns:
            DataFlowGraph: The optimized data flow graph after constant folding.
        """
        return graph

    def apply_accessed_fields(self, graph: DataFlowGraph) -> None:
        """Restrict the source feature type to only the accessed fields.

        This function retrieves the portion of the source node's data type that is
        explicitly accessed or utilized within the data flow graph. It represents
        the subset of features or fields from the source data type that are directly
        referenced by downstream nodes in the graph. Any features in the source data
        type that are not accessed remain excluded from this subset.

        The source feature of the data flow graph is updated to the accessed sub-feature.

        Args:
            graph (DataFlowGraph): The data flow graph to be optimized.
        """

        def get_accessed_dtype(node_id: NodeId) -> DType:
            source_dtype = graph.nodes[node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
            # only apply to non-leaf nodes and to mapping types
            if (graph.out_degree(node_id) == 0) or (not isinstance(source_dtype, MappingType)):
                return source_dtype

            accessed_fields = {}

            for _, v in graph.out_edges(node_id):
                node = graph.nodes[v][DataFlowGraph.NodeAttribute.NODE_OBJ]
                node_type = graph.nodes[v][DataFlowGraph.NodeAttribute.NODE_TYPE]

                if node_type != DataFlowGraph.NodeType.DATA_PROCESSOR:
                    return graph.nodes[node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]

                if isinstance(node, MappingGetItem):
                    # only a sub-feature of the mapping is accessed
                    accessed_fields[node.config.key] = get_accessed_dtype(v)
                else:
                    # the feature is used as a whole
                    return source_dtype

            return MappingType.construct(accessed_fields)

        graph.nodes[graph.src_node_id][
            DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE
        ] = get_accessed_dtype(graph.src_node_id)

    def optimize(self, graph: DataFlowGraph, leaf_nodes: set[NodeId]) -> DataFlowGraph:
        """Optimizes the data flow graph for a specified set of leaf nodes.

        Args:
            graph (DataFlowGraph): The data flow graph.
            leaf_nodes (set[NodeId]): Set of leaf node IDs.

        Returns:
            DataFlowGraph: The optimized data flow graph.

        Raises:
            AssertionError: If not all leaf nodes are contained in the optimized graph.
        """
        # build dependency graph for the given set of nodes
        # and always include the source node
        graph = graph.dependency_graph(leaf_nodes | {graph.src_node_id})

        # evaluate all constants
        graph = self.constant_evaluation(graph, leaf_nodes)

        # apply common sub-expresison elimination
        graph, _ = self.cse(graph)

        # apply constant folding/propagation
        graph = self.constant_folding(graph)

        # recompute all depths after optimiztion
        graph.recompute_depths()

        # limit the source features to only the accessed source features
        self.apply_accessed_fields(graph)

        # make sure all leaf nodes are present in the optimized graph
        assert all(node_id in graph for node_id in leaf_nodes)

        return graph
