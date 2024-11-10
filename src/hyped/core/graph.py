"""Defines the structure of data flow graphs and their components.

This module provides the :class:`DataFlowGraph` class and related components, which represent
the structure of a data processing workflow. The graph consists of nodes (data processors)
and edges (data flow between processors).
"""


from __future__ import annotations

import operator
import uuid
from enum import Enum
from functools import wraps
from itertools import groupby
from typing import Any, Hashable

import networkx as nx
import pyarrow as pa

from hyped.common.typing import NodeId, PartitionId

from .abstract import AbstractDataFlowGraph
from .features.engine import FeatureEngine
from .features.features import _Feature, build_feature_from_dtype
from .features.reference import FeatureKey, Reference
from .features.types import MappingType, SequenceType, Type
from .nodes.aggregator import BaseDataAggregator
from .nodes.augmenter import BaseDataAugmenter
from .nodes.base import BaseNode
from .nodes.collect import CollectNode
from .nodes.processor import BaseDataProcessor
from .utils import NestedType, build_dtype_from_object, map_recursive


def _compute_node_depth(G: nx.DiGraph) -> dict[Hashable, int]:
    """Compute the depth of each node in the graph.

    This function calculates the depth for each node in the provided networkx
    graph, where depth is defined as the shortest path length from a designated
    root node to each node in the graph. If the graph is a directed acyclic graph
    (DAG), the root is typically a node with no incoming edges.

    Args:
        G (nx.Graph): The networkx graph for which to compute node depths.

    Returns:
        dict: A dictionary mapping each node to its depth in the graph.

    Raises:
        ValueError: If the graph contains cycles.
        ValueError: If a root node cannot be determined.
    """
    node_depths = {}
    # trafers graph in topological order and compute the node depth
    for partition_node in nx.topological_sort(G):
        node_depths[partition_node] = max(
            (node_depths[parent] + 1 for parent in G.predecessors(partition_node)),
            default=0,
        )

    return node_depths


def _build_dependency_graph(
    G: nx.DiGraph, nodes: set[Hashable], stop_nodes: set[Hashable] = set()
) -> nx.DiGraph:
    """Build a subgraph containing all dependencies for a given set of nodes.

    This function constructs a subgraph from a directed graph :code:`G` by including
    all nodes that are dependencies (predecessors) of the specified :code:`nodes`,
    except those that reach any node in :code:`stop_nodes`. The resulting subgraph
    consists of the nodes in :code:`nodes` and all their upstream dependencies, but
    traversal stops at nodes in :code:`stop_nodes`.

    Args:
        G (nx.DiGraph): The original directed graph.
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
    assert all(node in G for node in nodes), "All nodes must be present in the graph 'G'."

    visited = set()
    to_visit = nodes.copy()

    # Traverse dependencies
    while to_visit:
        node = to_visit.pop()
        if node not in visited:
            visited.add(node)
            # Only add predecessors if the current node is not a stop node
            if node not in stop_nodes:
                to_visit.update(pred for pred in G.predecessors(node) if pred not in visited)

    return G.subgraph(visited)


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

        CONST = "CONSTANT"
        """
        Represents the partition containing all constant nodes.

        This partition includes nodes that hold constant values used in the
        data processing flow. This convers the actual constant nodes introducing
        constant values to the flow, as well as computations on only constant
        values.
        """

        DEFAULT = "DEFAULT"
        """
        Represents the default partition for nodes.

        This partition is assigned to the source node and is inherited by
        its sub-graph.
        """

        AGGREGATED = "AGGREGATED"
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

        COLLECT = "COLLECT_NODE"

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

        DATA_AUGMENTER = "DATA_AUGMENTER_NODE"
        """
        Represents a data augmenter node in the data flow graph.

        This type of node is responsible for modifying the dataset by generating
        new samples from existing ones or filtering out certain samples. Data
        augmenter nodes are used to expand or contract the dataset.
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

        Type: :class:`Type`

        This property contains the output feature produced by the data processor,
        as a :code:`Type` instance. It defines the structure and types of data that
        is generated by this node.
        """

        PARTITION = "partition"
        """
        Represents the partition to which the node belongs.

        Type: :class:`PartitionId`

        This property indicates the specific partition of the data flow graph that
        the node is part of, which can be used to group nodes by different semantics.
        """

        DEPTH = "depth"
        """
        Represents the depth of the node within the data flow graph.

        Type: :class:`int`

        This property indicates the level of the node in the graph, with the root
        node having a depth of 0. It is used to understand the hierarchical position
        of the node relative to other nodes in the data flow.
        """

    class EdgeAttribute(str, Enum):
        """Enum representing properties of an edge in the data flow graph."""

        NAME = "name"
        """
        Represents the name of the edge.

        Type: :class:`str`

        This property corresponds to the keyword of the argument used as an input
        to the processor, linking the edge to a specific input parameter.

        The name is also used by NetworkX as an identifier to distinguish multiedges
        between a pair of nodes. It serves as a unique identifier for the edge.
        """

        KEY = "feature_key"
        """
        Represents the key of the feature associated with the edge.

        Type: :class:`FeatureKey`

        This property specifies which subfeature of the output of the source node
        is flowing through the edge. It defines the particular feature that is being
        transmitted from one node to another in the data flow graph.
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

    @property
    def src_node_id(self) -> NodeId:
        """Get the source node ID of the data flow graph.

        This property returns the source node ID associated with the data flow graph.
        The source node is the entrypoint for inputs to the data flow.

        Returns:
            NodeId: The uuid of the source node.
        """
        return self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID]

    @property
    def src_dtype(self) -> Type:
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

    def build_partition_graph(self) -> nx.DiGraph:
        """Construct a partition graph from the data flow graph.

        This method builds a directed graph where each node represents a partition
        within the data flow graph, and edges represent the flow of data between
        these partitions. The resulting partition graph is required to have a tree
        structure, where each partition (except the root) has a single parent partition.

        Note that while the partition graph includes the constant partition, it does not
        model the flow of constants to other partitions, i.e. the constant partition is
        not isolated from the remaining partition graph. Reason for this design choice is
        that constants are special in that they can be used in any partition.

        Returns:
            nx.DiGraph: A directed graph representing the partitioned data flow.

        Raises:
            AssertionError: If the resulting partition graph is not a tree (i.e., any node
                            has more than one incoming edge).
        """
        G = nx.DiGraph()
        G.add_nodes_from(
            [
                DataFlowGraph.Partition.CONST.value,
                DataFlowGraph.Partition.DEFAULT.value,
            ]
        )

        # TODO: the aggregated partition is a special case since multiple
        #       independent partitions can point into it, this is not captured
        #       in the partition graph yet and would also break the tree structure
        #       asserted below
        for node_id, attrs in self.nodes(data=True):
            src_partition = attrs[DataFlowGraph.NodeAttribute.PARTITION]
            tgt_partition = self.get_node_output_partition(node_id)

            # make sure the source partition is contained in the partition graph
            assert (
                src_partition in G
            ), f"The source partition '{src_partition}' is not present in the partition graph."

            # don't include edges from the constant partition in partition graph
            if src_partition == DataFlowGraph.Partition.CONST:
                continue

            # add the target partition to the partition graph
            if tgt_partition not in G:
                G.add_node(tgt_partition)

            # only add a connection if there is no path from the source to the
            # target partition yet
            if (src_partition != tgt_partition) and not nx.has_path(
                G, src_partition, tgt_partition
            ):
                G.add_edge(src_partition, tgt_partition)

        # the partition graph needs to be a tree structure
        assert max(dict(G.in_degree).values()) <= 1, (
            "The partition graph must be a tree structure, but a node with more "
            "than one incoming edge was found."
        )

        return G

    def add_node(
        self,
        node_obj: Any,
        node_type: DataFlowGraph.NodeType,
        inputs: dict[str, Reference],
        output_type: Type,
        node_id: None | NodeId = None,
    ) -> Reference:
        # infer the output partition of the node from the node type and input references
        partition = self.infer_node_partition(node_type, list(inputs.values()))
        # aggregated partition currently only supports processor type nodes
        if (partition == DataFlowGraph.Partition.AGGREGATED) and (
            node_type not in {DataFlowGraph.NodeType.COLLECT, DataFlowGraph.NodeType.DATA_PROCESSOR}
        ):
            raise NotImplementedError(
                f"Aggregator outputs may only be processed by data processors "
                f"or collect operations, got {node_type}."
            )

        # compute the depth of the node in the graph based on it's input references
        depth = (
            0
            if inputs is None
            else max(
                (
                    self.nodes[ref._node_id][DataFlowGraph.NodeAttribute.DEPTH] + 1
                    for ref in inputs.values()
                ),
                default=0,
            )
        )

        # build the input data type from the input references
        input_type = MappingType.from_dict(
            {key: self.get_dtype_from_reference(ref) for key, ref in inputs.items()}
        )

        # create a random node id if no was given
        node_id = node_id if node_id is not None else str(uuid.uuid4())
        # add the node to the graph
        super(DataFlowGraph, self).add_node(
            node_id,
            **{
                DataFlowGraph.NodeAttribute.NODE_OBJ: node_obj,
                DataFlowGraph.NodeAttribute.NODE_TYPE: node_type,
                DataFlowGraph.NodeAttribute.IN_FEATURE_TYPE: input_type,
                DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE: output_type,
                DataFlowGraph.NodeAttribute.PARTITION: partition,
                DataFlowGraph.NodeAttribute.DEPTH: depth,
            },
        )

        # add dependency edges to graph
        for name, ref in inputs.items():
            # add the edge
            self.add_edge(
                ref._node_id,
                node_id,
                key=name,
                **{
                    DataFlowGraph.EdgeAttribute.NAME: name,
                    DataFlowGraph.EdgeAttribute.KEY: ref._key,
                },
            )

        # make sure the graph is a DAG
        assert nx.is_directed_acyclic_graph(self)

        return Reference(FeatureKey(), node_id, self)

    def add_source_node(self, data_type: Type, node_id: None | NodeId = None) -> Reference:
        """Add a the source node to the graph.

        This method adds a source node to the graph, which acts as the initial
        data provider for the data flow.

        Args:
            feature_type (Type): The feature type of the source node.
            node_id (None | NodeId): The id of the node, defaults to a random uuid.

        Returns:
            Reference: A reference object to the source node.

        Raises:
            AssertionError: If the graph already contains a source node
        """
        # make sure the graph has no source node yet
        if self.src_node_id is not None:
            raise RuntimeError("Graph already contains a source node.")

        # add the node to the graph
        ref = self.add_node(
            node_obj=None,
            node_type=DataFlowGraph.NodeType.SOURCE,
            inputs={},
            output_type=data_type,
            node_id=node_id,
        )
        # save source node id as graph attribute
        self.graph[DataFlowGraph.GraphAttribute.SRC_NODE_ID] = ref._node_id
        # return the reference to the source node
        return ref

    def add_const_node(self, value: Any, dtype: Type, node_id: None | NodeId = None) -> Reference:
        # make sure the data type matches the value
        array = pa.array([value], type=dtype.arrow_type)
        # create a random node id if not provided
        node_id = node_id if node_id is not None else str(uuid.uuid4())

        # add the node to the graph
        return self.add_node(
            node_obj=array,
            node_type=DataFlowGraph.NodeType.CONST,
            inputs={},
            output_type=dtype,
            node_id=node_id,
        )

    def add_collect_node(
        self, collect: NestedType[Reference | str | int | float], node_id: None | NodeId = None
    ) -> Reference:
        def add_constants(
            val: NestedType[Reference | str | int | float], dtype: None | Type = None
        ) -> NestedType[Reference]:
            if isinstance(val, dict):
                assert (dtype is None) or isinstance(dtype, MappingType)

                return {
                    key: add_constants(item, dtype[key] if dtype is not None else None)
                    for key, item in val.items()
                }

            elif isinstance(val, (list, tuple)):
                assert (dtype is None) or isinstance(dtype, SequenceType)

                if len(val) == 0:
                    raise NotImplementedError("Empty Sequence")

                if dtype is None:
                    # try to infer the dtype from the reference instances in the sequence
                    if any(isinstance(r, Reference) for r in val):
                        ref = next(r for r in val if isinstance(r, Reference))
                        dtype = self.get_dtype_from_reference(ref)

                else:
                    # otherwise use the value type from the given dtype
                    dtype = dtype.value_type

                # recurse on all items in the sequence
                return type(val)([add_constants(item, dtype) for item in val])

            elif not isinstance(val, Reference):
                # add the constant node
                return self.add_const_node(
                    val, dtype=dtype if dtype is not None else build_dtype_from_object(val)
                )

            elif isinstance(val, Reference):
                return val

            else:
                # TODO: error message
                TypeError(val)

        if isinstance(collect, Reference):
            # nothing to collect
            return collect

        # prepare collect structure and add all included constants to the graph
        collect = map_recursive(lambda _, x: x.ref if isinstance(x, _Feature) else x, collect)
        collect = add_constants(collect)

        inputs = {}
        # extract flat inputs to collect node from structure
        map_recursive(
            lambda p, r: (
                None
                if not isinstance(r, Reference)
                else operator.setitem(inputs, ".".join(map(str, p)), r)
            ),
            collect,
        )

        # build the nested value lookup structure
        lookup = map_recursive(
            lambda p, v: ".".join(map(str, p)) if isinstance(v, Reference) else v, collect
        )

        # create the collect node object
        obj = CollectNode(lookup=lookup)

        # add the node object
        return self.add_node(
            node_obj=obj,
            node_type=DataFlowGraph.NodeType.COLLECT,
            inputs=inputs,
            output_type=obj.build_output_type(self, inputs),
            node_id=node_id,
        )

    # TODO: rename to more generic 'add_node'
    def add_processor_node(
        self,
        obj: BaseNode,
        inputs: dict[str, Reference],
        node_id: None | NodeId = None,
    ) -> Reference:
        """Add a processor node to the data flow graph.

        This method adds a processor node to the data flow graph and creates the
        necessary edges to define the data flow from input nodes to this processor.

        Note that this function does not do any input type validation as this is implemented
        in the call method of the node object.

        Args:
            obj (BaseNode): The node object.
            inputs (dict[str, Reference]): The input references to the node.
            node_id (None | NodeId): The id of the node, defaults to a random uuid.

        Returns:
            Reference: A reference instance to the added node.

        Raises:
            AssertionError: If the processor type is invalid.
            AssertionError: If the graph is cyclic after adding the new node.
            AssertionError: If the partition cannot be inferred.
            RuntimeError: If any input reference do not belong to this data flow.
            RuntimeError: If the input references are a mix of aggregated and non-aggregated
                features.
        """
        # get processor type
        node_type = (
            DataFlowGraph.NodeType.SOURCE
            if obj is None
            else DataFlowGraph.NodeType.DATA_PROCESSOR
            if isinstance(obj, BaseDataProcessor)
            else DataFlowGraph.NodeType.DATA_AGGREGATOR
            if isinstance(obj, BaseDataAggregator)
            else DataFlowGraph.NodeType.DATA_AUGMENTER
            if isinstance(obj, BaseDataAugmenter)
            else None
        )
        # make sure the object is valid
        assert node_type is not None, f"Invalid node type {type(obj)}."

        # make sure all input references belong to this graph
        if (inputs is not None) and any(ref._graph is not self for ref in inputs.values()):
            raise RuntimeError("Input reference does not belong to this data flow graph.")

        # build the input features
        input_features = {key: self.get_feature_from_reference(ref) for key, ref in inputs.items()}
        # create a type validation engine instance
        name = f"DataFlowGraph.add_node({type(obj).__qualname__})"
        engine = FeatureEngine(name, obj.config, obj.signature)
        # validate the input features to the node
        engine.validate_signature()
        engine.validate_arguments(**input_features)
        # get the output feature type of the node for the given inputs
        feature_type = engine.build_return_feature(Reference(), input_features).dtype

        return self.add_node(
            node_obj=obj,
            node_type=node_type,
            inputs=inputs,
            output_type=feature_type,
            node_id=node_id,
        )

    def get_node_output_partition(self, node_id: NodeId) -> PartitionId:
        """Determine the output partition for a given node in the data flow graph.

        This method determines the partition that the output of a specified node
        belongs to, based on the node's type. Different types of nodes may direct
        their output to different partitions, reflecting their role in the data flow:

        - :code:`DATA_AGGREGATOR`: Outputs always point to the :code:`AGGREGATED` partition,
          even though the aggregator node itself is not part of this partition.

        - :code:`DATA_AUGMENTER`: Outputs always point to their own partition, using the
          node's ID as the partition name. Like aggregators, augmenters are not part of
          the partition they point to.

        - Other node types: Outputs remain within the partition specified by the
          node's :code:`PARTITION` attribute.

        Args:
            node_id (NodeId): The ID of the node for which to determine the output partition.

        Returns:
            PartitionId: The partition that the node's output will be directed to.
        """
        input_node = self.nodes[node_id]
        input_node_type = input_node[DataFlowGraph.NodeAttribute.NODE_TYPE]

        if input_node_type == DataFlowGraph.NodeType.DATA_AGGREGATOR:
            # data aggregator outputs always point into
            # the aggregated partition while the aggregator
            # node itself is not part of the aggregated partition
            return DataFlowGraph.Partition.AGGREGATED.value

        elif input_node_type == DataFlowGraph.NodeType.DATA_AUGMENTER:
            # data augmenters always point into their own partition
            # we re-use the node-id of the augmenter as the partition
            # note that while they point into their own partition, they
            # are not part of them, similar to aggregators they point
            # from outside the partition into it
            return node_id

        else:
            # other node types don't transition between partitions
            return input_node[DataFlowGraph.NodeAttribute.PARTITION]

    def infer_node_partition(
        self, node_type: DataFlowGraph.NodeType, refs: list[Reference]
    ) -> PartitionId:
        """Infer the appropriate partition for a given node.

        This method determines the partition to which a node belongs, based on its
        type and the partitions of its input references.

        The method handles different node types as follows:
        - :code:`SOURCE`: Assigned to the default partition.
        - :code:`CONST`: Assigned to the constant partition.
        - Other node types: The partition is inferred based on the partitions of
        the input references.

        The partition is inferred from the inputs as follows:
        - If the input references are from constant partitions, the node is also assigned
        to the constant partition.
        - If any input comes from an aggregated partition, the node is assigned to the
        aggregated partition.
        - If multiple candidate partitions are found, the method selects the partition
        that is deepest in the partition graph.

        If the resulting partitions include a mix of independent partitions, an error is raised.
        Independent partitions are those that are not connected in the partition graph, meaning
        the resulting partition graph would not form a valid tree structure. This ensures that
        the partition graph remains a tree, with each partition having a single parent and no
        cycles or disjoint components.

        Args:
            node_type (DataFlowGraph.NodeType): The type of the node for which to infer
                the partition.
            refs (list[Reference]): A list of input references associated with the node.

        Returns:
            PartitionId: The inferred partition for the node.

        Raises:
            AssertionError: If no input references are provided.
            RuntimeError: If conflicting partitions are mixed.
        """
        if node_type == DataFlowGraph.NodeType.SOURCE:
            # source node is added to the default partition
            return DataFlowGraph.Partition.DEFAULT.value

        if node_type == DataFlowGraph.NodeType.CONST:
            # contants are added to the constant partition
            return DataFlowGraph.Partition.CONST.value

        # partition could not be inferred
        assert len(refs) > 0, "Partition cannot be inferred for nodes without any input references."

        # get the input partitions
        candidate_partitions = {self.get_node_output_partition(ref._node_id) for ref in refs}

        if candidate_partitions == {DataFlowGraph.Partition.CONST}:
            # if all inputs come from the constant partition, then this node
            # is also part of the constant partition
            return DataFlowGraph.Partition.CONST.value

        if DataFlowGraph.Partition.AGGREGATED in candidate_partitions:
            # if the inputs come directly from an aggregator or from the
            # aggregated partition, then stay in the aggregated partition

            if (
                len(
                    candidate_partitions
                    - {
                        DataFlowGraph.Partition.AGGREGATED,
                        DataFlowGraph.Partition.CONST,
                    }
                )
                != 0
            ):
                raise RuntimeError("Cannot mix aggregated and non-aggregated features.")

            return DataFlowGraph.Partition.AGGREGATED.value

        # remove the constant partition from the set of candidates
        # if there is any other partition to select from
        candidate_partitions -= {DataFlowGraph.Partition.CONST}

        if len(candidate_partitions) == 1:
            return next(iter(candidate_partitions))

        # build the partition graph and compute the depth of each node
        p_graph = self.build_partition_graph()
        p_depths = _compute_node_depth(p_graph)

        # select the candidate partition that is deepest in the partition graph
        # as it is the only one that can consume all candidate partitions
        candidate = max(candidate_partitions, key=p_depths.get)

        p_dep_graph = _build_dependency_graph(p_graph, {candidate})
        # build the set of allowed partitions
        # note that the constant partition is always allowed
        allowed_partitions = set(p_dep_graph.nodes())
        allowed_partitions.add(DataFlowGraph.Partition.CONST.value)
        # make sure only allowed partitions are used
        if not candidate_partitions.issubset(allowed_partitions):
            raise RuntimeError("Cannot mix independent partitions.")

        return candidate

    def get_dtype_from_reference(self, ref: Reference) -> Type:
        """Helper function to get the data type of a referenced feature.

        Args:
            ref (Reference): The reference for which to get the type.

        Returns:
            Type: The data type of the referenced feature.

        Raises:
            RuntimeError: If the reference's node ID is not contained in the graph.
        """
        if ref._node_id not in self.nodes:
            raise RuntimeError(f"Node with ID '{ref._node_id}' is not contained in the graph.")

        # get the output type of the referenced node
        dtype = self.nodes[ref._node_id][DataFlowGraph.NodeAttribute.OUT_FEATURE_TYPE]
        return ref._key.index_dtype(dtype)

    def get_feature_from_reference(self, ref: Reference) -> _Feature:
        """Helper function to get a feature instance of some node output.

        Args:
            ref (Reference): The reference instance indicating the node
                generating the feature and the specific sub-feature.

        Returns:
            _Feature: The arrow type of the referenced feature.

        Raises:
            RuntimeError: If the reference's node ID is not contained in the graph.
        """
        # build the feature instance from the reference and data type
        dtype = self.get_dtype_from_reference(ref)
        return build_feature_from_dtype(ref, dtype)

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

    def subgraph_in_edges(
        self, subgraph: DataFlowGraph, data: bool | EdgeAttribute = False
    ) -> list[tuple[int, int, str] | tuple[int, int, str, Any]]:
        """Get incoming edges to a subgraph from nodes outside the subgraph.

        This method returns a list of edges that point to nodes within the
        specified subgraph from nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.
            data (bool | EdgeAttribute): Whether to include edge data. If set to
                :code:`True` or an :class:`EdgeAttribute`, the method returns edges with data.

        Returns:
            list[tuple[int, int, str] | tuple[int, int, str, Any]]: The incoming edges
            to the subgraph.
        """
        return [e for e in self.in_edges(subgraph, keys=True, data=data) if e[0] not in subgraph]

    def subgraph_out_edges(
        self, subgraph: DataFlowGraph, data: bool | EdgeAttribute = False
    ) -> list[tuple[int, int, str] | tuple[int, int, str, Any]]:
        """Get outgoing edges from a subgraph to nodes outside the subgraph.

        This method returns a list of edges that point from nodes within the
        specified subgraph to nodes outside the subgraph.

        Args:
            subgraph (DataFlowGraph): The subgraph of interest.
            data (bool | EdgeAttribute): Whether to include edge data. If set to
                :code:`True` or an :class:`EdgeAttribute`, the method returns edges with data.

        Returns:
            list[tuple[int, int, str] | tuple[int, int, str, Any]]: The outgoing edges
            from the subgraph.
        """
        return [e for e in self.out_edges(subgraph, keys=True, data=data) if e[1] not in subgraph]

    def recompute_depths(self) -> None:
        """Recompute the depth of all nodes in the data flow graph.

        This method recalculates the depth of each node based on the topological
        order of the graph. The depth of a node is defined as the length of the
        longest path from the source node to the node.
        """
        node_depths = _compute_node_depth(self)
        nx.set_node_attributes(self, node_depths, DataFlowGraph.NodeAttribute.DEPTH)
