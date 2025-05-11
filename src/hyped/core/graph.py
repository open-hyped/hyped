"""Defines the structure of data flow graphs and their components.

This module provides the :class:`DataFlowGraph` class and related components, which represent
the structure of a data processing workflow. The graph consists of nodes (data processors)
and edges (data flow between processors).
"""


from __future__ import annotations

from enum import Enum
from functools import wraps
from itertools import groupby
from typing import Any, Hashable

import networkx as nx
from jinja2 import Template

from .abc import AbstractDataFlowGraph
from .features.dtypes import DType, MappingType, build_dtype_from_dict
from .registry.config import AutoConfigurable
from .typing import NodeId, PartitionId
from .utils import random_uuid

DEFAULT_NODE_FORMAT = (
    "[{{ node_id[:4] }}] "
    "{% if node_type == 'SOURCE_NODE' %}"
    "Source"
    "{% else %}"
    "{{ node_object }}"
    "{% endif %}"
)


def _compute_node_depth(g: nx.DiGraph) -> dict[Hashable, int]:
    """Compute the depth of each node in the graph.

    This function calculates the depth for each node in the provided networkx
    graph, where depth is defined as the shortest path length from a designated
    root node to each node in the graph. If the graph is a directed acyclic graph
    (DAG), the root is typically a node with no incoming edges.

    Args:
        g (nx.Graph): The networkx graph for which to compute node depths.

    Returns:
        dict: A dictionary mapping each node to its depth in the graph.

    Raises:
        ValueError: If the graph contains cycles.
        ValueError: If a root node cannot be determined.
    """
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Graph contains a cycle.")

    node_depths = {}
    # trafers graph in topological order and compute the node depth
    for partition_node in nx.topological_sort(g):
        node_depths[partition_node] = max(
            (node_depths[parent] + 1 for parent in g.predecessors(partition_node)),
            default=0,
        )

    return node_depths


def _build_dependency_graph(
    g: nx.DiGraph, nodes: set[Hashable], stop_nodes: set[Hashable] = set()
) -> nx.DiGraph:
    """Build a subgraph containing all dependencies for a given set of nodes.

    This function constructs a subgraph from a directed graph :code:`G` by including
    all nodes that are dependencies (predecessors) of the specified :code:`nodes`,
    except those that reach any node in :code:`stop_nodes`. The resulting subgraph
    consists of the nodes in :code:`nodes` and all their upstream dependencies, but
    traversal stops at nodes in :code:`stop_nodes`.

    Args:
        g (nx.DiGraph): The original directed graph.
        nodes (set[Hashable]): A set of nodes for which to build the dependency subgraph.
        stop_nodes (set[Hashable]): A set of nodes that serve as cut-off points in
            the traversal. Dependencies beyond these nodes will not be included
            in the subgraph.

    Returns:
        nx.DiGraph: A subgraph of :code:`G` containing the specified nodes and their dependencies,
        with paths beyond :code:`stop_nodes` excluded.

    Raises:
        AssertionError: If any node in :code:`nodes` is not present in :code:`G`.
    """
    assert all(node in g for node in nodes), "All nodes must be present in the graph 'G'."

    visited = set()
    to_visit = nodes.copy()

    # Traverse dependencies
    while to_visit:
        node = to_visit.pop()
        if node not in visited:
            visited.add(node)
            # Only add predecessors if the current node is not a stop node
            if node not in stop_nodes:
                to_visit.update(pred for pred in g.predecessors(node) if pred not in visited)

    return g.subgraph(visited)


class DataFlowGraph(nx.MultiDiGraph, AbstractDataFlowGraph):
    """A multi-directed graph representing a data flow of data processors.

    This class is used internally to define a directed acyclic graph (DAG)
    where nodes represent data processors of `BaseDataProcessor` type, and
    edges define the data flow between these processors.
    """

    class GraphAttribute(str, Enum):
        """Enum representing properties of the data flow graph."""

        SRC_NODE_ID = "src_node_id"
        """
        Property representing the source node ID.

        This property identifies the source node ID of an edge in the graph.
        """

    class Partition(str, Enum):
        """Enum representing predefined partitions in the data flow graph."""

        CONST = str(random_uuid())
        """
        Represents the partition containing all constant nodes.

        This partition includes nodes that hold constant values used in the
        data processing flow. This convers the actual constant nodes introducing
        constant values to the flow, as well as computations on only constant
        values.
        """

        DEFAULT = str(random_uuid())
        """
        Represents the default partition for nodes.

        This partition is assigned to the source node and is inherited by
        its sub-graph.
        """

        AGGREGATED = str(random_uuid())
        """
        Represents the partition containing aggregated values.

        This partition includes all aggregated values in a data flow, i.e.
        nodes that process the output of aggregator nodes.
        """

    class NodeType(str, Enum):
        """Enum representing types of nodes in the data flow graph."""

        SOURCE = "SOURCE_NODE"
        """
        Represents a source node in the data flow graph.

        This type of node acts as the starting point of the data flow graph,
        typically representing raw input data sources.
        """

        CONST = "CONST_NODE"
        """
        Represents a constant node in the data flow graph.

        This type of node introduces constant values into the data flow, serving
        as fixed inputs to the subsequent processing stages.
        """

        CAST = "CAST_NODE"
        """
        Represents a cast node in the data flow graph.

        This type of node is responsible for type conversion within the data flow.
        It transforms data from one type to another, ensuring compatibility between
        different nodes or preparing the data for specific processing requirements.
        """

        COLLECT = "COLLECT_NODE"
        """
        Represents a collect node in the data flow graph.

        This type of node collects features from multiple upstream nodes into a single
        (nested) feature. This enables the combination of outputs from various sources
        or transformations into a unified structure, which can be further processed
        downstream.
        """

        TRACE = "TRACE_NODE"
        """
        Represents a trace node in the data flow graph.

        This type of node is responsible for tracing values through different
        partitions of the graph. It transforms data of a specific partition into
        the index-space of a target partition.
        """

        DEBUG = "DEBUG_NODE"
        """Represents a debug node for inspecting data flow.

        Debug nodes do not transform data or produce integrated outputs. They
        facilitate monitoring and debugging through mechanisms like logging or
        asserting.
        """

        DATA_PROCESSOR = "DATA_PROCESSOR_NODE"
        """
        Represents a data processor node in the data flow graph.

        This type of node represents a data processing component within the
        data flow graph. Data processors perform specific transformations
        on input data and produce output data based on defined processing logic.
        """

        DATA_AGGREGATOR = "DATA_AGGREGATOR_NODE"
        """
        Represents a data aggregator node in the data flow graph.

        This type of node is responsible for aggregating samples. Aggregator
        nodes typically perform dataset-wide computations.
        """

        DATA_AUGMENTOR = "DATA_AUGMENTOR_NODE"
        """
        Represents a data augmentor node in the data flow graph.

        This type of node is responsible for modifying the dataset by generating
        new samples from existing ones or filtering out certain samples. Data
        augmentor nodes are used to expand or contract the dataset.
        """

    class NodeAttribute(str, Enum):
        """Enum representing properties of a node in the data flow graph."""

        NODE_OBJ = "node_object"
        """
        The object associated with the node.

        The value of this property is dependent on the type of node. For
        nodes of type :class:`NodeType.DATA_PROCESSOR`, this property
        refers to the processor instance of the node.
        """

        NODE_TYPE = "node_type"
        """
        Indicates the type of node.

        Type: :class:`NodeType`

        This property indicates the type of data processor. It helps in categorizing
        and identifying the nature of the processor in the data flow graph.
        """

        IN_FEATURE_TYPE = "in_feature_type"
        """
        Represents the input features associated with the node.

        Type: :class:`MappingType`

        This property contains the input features required by the data processor,
        in the form of a mapping type instance. It defines the structure and types
        of the inputs expected by the node.
        """

        OUT_FEATURE_TYPE = "out_feature_type"
        """
        Represents the output features associated with the node.

        Type: :class:`DType`

        This property contains the output feature produced by the data processor,
        as a :code:`DType` instance. It defines the structure and types of data that
        is generated by this node.
        """

        PARTITION = "partition"
        """
        Represents the partition to which the node belongs.

        Type: :class:`PartitionId`

        This property indicates the specific partition of the data flow graph that
        the node is part of, which can be used to group nodes by different semantics.
        """

        OUT_PARTITION = "out_partition"
        """
        Represents the partition the node points to.

        Type: :class:`PartitionId`

        This property indicates the specific partition of the data flow graph that
        the node output belongs to.
        """

        DEPTH = "depth"
        """
        Represents the depth of the node within the data flow graph.

        Type: :class:`int`

        This property indicates the level of the node in the graph, with the root
        node having a depth of 0. It is used to understand the hierarchical position
        of the node relative to other nodes in the data flow.
        """

    @wraps(nx.MultiDiGraph)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DataFlowGraph.

        Args:
            *args: Positional arguments forwarded to the init of the :class:`MultiDiGraph`.
            **kwargs: Keyword arguments forwarded to the init of the :class:`MultiDiGraph`.
        """
        super(DataFlowGraph, self).__init__(*args, **kwargs)

        # set default source node id
        if DataFlowGraph.GraphAttribute.SRC_NODE_ID not in self.graph:
            self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID] = None

        # reset source node id for subgraphs
        if (self.src_node_id is not None) and (self.src_node_id not in self):
            self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID] = None

    def format(self, format: str | Template = DEFAULT_NODE_FORMAT) -> nx.MultiDiGraph:
        """Creates a formatted copy with node labels generated using a Jinja2 template.

        This method applies a user-specified format to the attributes of each node in the graph,
        producing a new graph where each node has a :code:`label` attribute rendered from the
        provided template. The edges in the graph are copied without modification.

        Attributes available in the template are specified by the :class:`NodeAttribute` enum.

        Parameters:
            format (str | Template): A Jinja2 template string or a precompiled Jinja2
                :class:`Template` object. The template can reference any of the node
                attributes, as well as the :code:`node_id`.
                Defaults to :code:`DEFAULT_NODE_FORMAT`.

        Returns:
            nx.MultiDiGraph: A new graph where nodes have a :code:`label` attribute generated
                from the provided template, and edges are identical to those in the original
                graph.
        """
        # compile the jinja template
        template = Template(format) if not isinstance(format, Template) else format
        # apply the template to each node in the graph
        format_graph = nx.MultiDiGraph()
        for node_id, attrs in self.nodes(data=True):
            format_graph.add_node(node_id, label=template.render(**attrs, node_id=node_id))
        # add all edges
        format_graph.add_edges_from(self.edges)
        return format_graph

    def to_string(
        self,
        format: str = DEFAULT_NODE_FORMAT,
        ascii_only: bool = True,
        vertical_chains: bool = True,
    ) -> str:
        """Generates a string representation of the graph.

        This method provides a textual representation of the graph, with nodes and edges displayed
        in a human-readable format. Nodes are labeled according to the specified format string, and
        additional options allow customization of the output style.

        Attributes available in the template are specified by the :class:`NodeAttribute` enum.

        Parameters:
            format (str): A Jinja2 template string used to generate labels for the nodes.
                The template can reference any of the node attributes defined by the
                :class:`NodeAttribute` enum, as well as the :code:`node_id`. Defaults to
                :code:`DEFAULT_NODE_FORMAT`.
            ascii_only (bool): If :code:`True`, the output will use only ASCII characters.
                Defaults to :code:`True`.
            vertical_chains (bool): If :code:`True`, the output will display chains of nodes
                in a vertical layout for better readability. Defaults to :code:`True`.

        Returns:
            str: A string representation of the graph, formatted according to the
            specified options.
        """
        lines = nx.generate_network_text(
            self.format(format),
            with_labels=True,
            ascii_only=ascii_only,
            vertical_chains=vertical_chains,
        )
        return "\n".join(lines)

    @property
    def src_node_id(self) -> None | NodeId:
        """Get the source node ID of the data flow graph.

        This property returns the source node ID associated with the data flow graph.
        The source node is the entrypoint for inputs to the data flow.

        Returns:
            None | NodeId: The uuid of the source node. :code:`None` if graph has no source node.
        """
        return self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID]

    @property
    def src_dtype(self) -> MappingType:
        """Get the source data type.

        This property retrieves the data type of the source node in the data flow graph.
        The source data type defines the structure and types of features expected by the
        source node, which serve as the initial inputs to the data flow graph.

        Returns:
            MappingType: The data type of the source node.
        """
        return self.nodes[self.src_node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]

    @property
    def depth(self) -> int:
        """Computes the total depth of the data flow graph.

        The depth is defined as the maximum level of any node in the graph, where the root
        node has a depth of 0. This property calculates the depth by finding the maximum
        depth attribute among all nodes in the graph.

        Returns:
            int: The total depth of the graph.
        """
        return max(nx.get_node_attributes(self, DataFlowGraph.NodeAttribute.DEPTH).values()) + 1

    @property
    def width(self) -> int:
        """Computes the width of the data flow graph.

        The width is defined as the maximum number of nodes present at any single depth level
        in the graph. This property calculates the width by grouping nodes by their depth and
        finding the largest group.

        Returns:
            int: The maximum width of the graph.
        """
        # group nodes by their layer
        depths = nx.get_node_attributes(self, DataFlowGraph.NodeAttribute.DEPTH)
        layers = groupby(sorted(self, key=depths.get), key=depths.get)
        # find larges layer in graph
        return max(len(list(layer)) for _, layer in layers)

    def get_output_dtype(self, node_id: NodeId) -> DType:
        """Helper function to get the output data type of a node.

        Args:
            node_id (NodeId): The id of the node.

        Returns:
            DType: The output data type of the node.

        Raises:
            RuntimeError: If the node id is not contained in the graph.
        """
        if node_id not in self.nodes:
            raise RuntimeError(f"Node with ID '{node_id}' is not contained in the graph.")
        # get the output type of the referenced node
        return self.nodes[node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]

    def add_node(
        self,
        node_obj: Any,
        node_type: DataFlowGraph.NodeType,
        inputs: dict[str, NodeId],
        output_dtype: DType,
        partition: PartitionId,
        out_partition: PartitionId,
        node_id: NodeId,
    ) -> NodeId:
        """Adds a new node to the data flow graph.

        This function creates a node in the graph with specified attributes such as
        type, inputs, and output data type. It ensures the graph remains a Directed
        Acyclic Graph (DAG) after adding the node. The function also adds the
        dependency edges between the new node and its input nodes.

        Args:
            node_obj (Any): The object associated with the node.
            node_type (DataFlowGraph.NodeType): The type of the node.
            inputs (dict[str, NodeOd]): A dictionary mapping input names to
                :class:`NodeId` that represent dependencies of this node on
                other nodes in the graph.
            partition (PartitionId): The partition of the node.
            out_partition (PartitionId): The output partition of the node.
            output_dtype (DType): The output data type produced by this node.
            node_id (None | NodeId): The unique identifier for the node.

        Returns:
            NodeId: The node id of the added node.
        """
        assert node_id not in self.nodes, f"Node id '{node_id}' already in use."
        # compute the depth of the node in the graph based on it's input references
        depth = (
            0
            if inputs is None
            else max(
                (
                    self.nodes[node_id][DataFlowGraph.NodeAttribute.DEPTH] + 1
                    for node_id in inputs.values()
                ),
                default=0,
            )
        )

        # build the input data type from the input references
        input_dtype = MappingType(
            tuple((key, self.get_output_dtype(node_id)) for key, node_id in inputs.items())
        )

        # add the node to the graph
        super(DataFlowGraph, self).add_node(
            node_id,
            **{
                DataFlowGraph.NodeAttribute.NODE_OBJ.value: node_obj,
                DataFlowGraph.NodeAttribute.NODE_TYPE.value: node_type,
                DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE.value: input_dtype,
                DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE.value: output_dtype,
                DataFlowGraph.NodeAttribute.PARTITION.value: partition,
                DataFlowGraph.NodeAttribute.OUT_PARTITION.value: out_partition,
                DataFlowGraph.NodeAttribute.DEPTH.value: depth,
            },
        )

        # add dependency edges to graph
        self.add_edges_from((in_node_id, node_id, name, {}) for name, in_node_id in inputs.items())

        # make sure the graph is a DAG
        assert nx.is_directed_acyclic_graph(self)

        return node_id

    def add_source_node(self, dtype: DType, node_id: NodeId) -> NodeId:
        """Add a the source node to the graph.

        This method adds a source node to the graph, which acts as the initial
        data provider for the data flow.

        Args:
            dtype (DType): The data type representing the source features.
            node_id (NodeId): The id of the node.

        Returns:
            NodeId: The node id of the added source node.

        Raises:
            RuntimeError: If the graph already contains a source node
        """
        # make sure the graph has no source node yet
        if self.src_node_id is not None:
            raise RuntimeError("Graph already contains a source node.")

        # add the node to the graph
        node_id = self.add_node(
            node_obj=None,
            node_type=DataFlowGraph.NodeType.SOURCE,
            inputs={},
            output_dtype=dtype,
            partition=DataFlowGraph.Partition.DEFAULT,
            out_partition=DataFlowGraph.Partition.DEFAULT,
            node_id=node_id,
        )
        # save source node id as graph attribute
        self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID] = node_id
        # return the node id of the source node
        return node_id

    def dependency_graph(
        self, nodes: set[NodeId], stop_nodes: set[NodeId] = set()
    ) -> DataFlowGraph:
        """Generate the dependency subgraph for a given set of nodes.

        This method generates a subgraph that includes all nodes that the specified
        :code:`nodes` depend on, either directly or indirectly, up to any defined cut-off
        points.

        Args:
            nodes (set[NodeId]): The node IDs for which to generate the dependency graph.
            stop_nodes (set[NodeId]): A set of node IDs that serve as cut-off points in
                the traversal. When a dependency chain reaches any node in :node:`stop_nodes`,
                it stops there, excluding that node's dependencies from the subgraph.
                This allows limiting the scope of the dependency graph by excluding
                deeper dependencies beyond these nodes.

        Returns:
            DataFlowGraph: A subgraph representing the dependencies of the specified
            :code:`nodes`, excluding paths beyond any nodes in :code:`stop_nodes`.
        """
        return _build_dependency_graph(self, nodes, stop_nodes)

    def get_partition(self, partition: PartitionId) -> DataFlowGraph:
        """Extract a subgraph containing only nodes from a specific partition.

        This method creates a subgraph from the current graph by selecting
        nodes that belong to a specified partition.

        Args:
            partition (PartitionId): The partition identifier.

        Returns:
            DataFlowGraph: The subgraph containing nodes of the specified partition.
        """
        # get all nodes to the provided partition
        partition = [
            i
            for i, data in self.nodes(data=True)
            if data[DataFlowGraph.NodeAttribute.PARTITION] == partition
        ]
        # build the sub-graph of only the provided partition
        return self.subgraph(partition)

    def drop_partition(self, partition: PartitionId) -> DataFlowGraph:
        """Drop a specified partition from the graph.

        This method creates a subgraph from the current graph by excluding
        nodes that belong to a specified partition.

        Args:
            partition (PartitionId): The partition identifier.

        Returns:
            DataFlowGraph: The subgraph excluding nodes of the specified partition.
        """
        # get all nodes to the provided partition
        remainder = [
            i
            for i, data in self.nodes(data=True)
            if data[DataFlowGraph.NodeAttribute.PARTITION] != partition
        ]
        # build the sub-graph of only the provided partition
        return self.subgraph(remainder)

    def subgraph_in_edges(self, subgraph: DataFlowGraph) -> list[tuple[int, int, str]]:
        """Get incoming edges to a subgraph from nodes outside the subgraph.

        This method returns a list of edges that point to nodes within the
        specified subgraph from nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.

        Returns:
            list[tuple[int, int, str]]: The incoming edges to the subgraph.
        """
        return [e for e in self.in_edges(subgraph, keys=True) if e[0] not in subgraph]

    def subgraph_out_edges(self, subgraph: DataFlowGraph) -> list[tuple[int, int, str]]:
        """Get outgoing edges from a subgraph to nodes outside the subgraph.

        This method returns a list of edges that point from nodes within the
        specified subgraph to nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.

        Returns:
            list[tuple[int, int, str]]: The outgoing edges from the subgraph.
        """
        return [e for e in self.out_edges(subgraph, keys=True) if e[1] not in subgraph]

    def recompute_depths(self) -> None:
        """Recompute the depth of all nodes in the data flow graph.

        This method recalculates the depth of each node based on the topological
        order of the graph. The depth of a node is defined as the length of the
        longest path from the source node to the node.
        """
        node_depths = _compute_node_depth(self)
        nx.set_node_attributes(self, node_depths, DataFlowGraph.NodeAttribute.DEPTH)

    def to_dict(self) -> dict:
        """Serializes the data flow graph into a dictionary representation.

        This method generates a dictionary that represents the entire data flow graph,
        including the nodes and edges. Each node is serialized based on its type,
        with special handling for constants, casts, and various data processing nodes.
        Additionally, the feature types for each node are serialized.

        Returns:
            dict: A dictionary representation of the data flow graph, including nodes
                with their serialized objects and feature types, and links between them.
        """
        # get dictionary representation of data flow graph
        data = nx.node_link_data(self, edges="edges")

        for node in data["nodes"]:
            # serialize node object
            obj = node[DataFlowGraph.NodeAttribute.NODE_OBJ]
            node[DataFlowGraph.NodeAttribute.NODE_OBJ] = (
                None if obj is None else obj.config.to_dict()
            )
            # serialize feature types
            out_dtype = node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
            out_dtype = None if out_dtype is None else out_dtype.to_dict()
            node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] = out_dtype
            node[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] = node[
                DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE
            ].to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> DataFlowGraph:
        """Deserializes a dictionary into a data flow graph.

        This method converts a dictionary (typically one generated by :code:`to_dict`)
        back into a :class:`DataFlowGraph` instance. It handles the deserialization of
        node types, partitions, feature types, and node objects. For each node, the
        method reconstructs its associated object based on the node's type. Additionally,
        edge data is deserialized and used to re-establish links between the nodes.

        Args:
            data (dict): A dictionary representation of a data flow graph, including
                node data and links between them.

        Returns:
            DataFlowGraph: The deserialized data flow graph.
        """
        for node in data["nodes"]:
            # deserialize node types
            node[DataFlowGraph.NodeAttribute.NODE_TYPE] = DataFlowGraph.NodeType(
                node[DataFlowGraph.NodeAttribute.NODE_TYPE]
            )
            # deserialize partition
            if node[DataFlowGraph.NodeAttribute.PARTITION] in set(DataFlowGraph.Partition):
                node[DataFlowGraph.NodeAttribute.PARTITION] = DataFlowGraph.Partition(
                    node[DataFlowGraph.NodeAttribute.PARTITION]
                )
            # deserialize feature types
            out_dtype = node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
            out_dtype = None if out_dtype is None else build_dtype_from_dict(out_dtype)
            node[DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE] = out_dtype
            node[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE] = build_dtype_from_dict(
                node[DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE]
            )
            # deserialize node objects
            obj = node[DataFlowGraph.NodeAttribute.NODE_OBJ]
            node[DataFlowGraph.NodeAttribute.NODE_OBJ] = (
                None if obj is None else AutoConfigurable.from_config_dict(obj)
            )

        return DataFlowGraph(nx.node_link_graph(data, edges="edges"))
